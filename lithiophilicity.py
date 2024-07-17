#!/usr/bin/env python3

import os

import dolfinx
import gmsh
import matplotlib.pyplot as plt
import meshio
import numpy as np
import pyvista
import pyvista as pv
import pyvistaqt as pvqt
import ufl
import warnings

from dolfinx import cpp, fem, io, mesh, nls, plot
from dolfinx.fem import petsc
from dolfinx.io import VTXWriter
from dolfinx.nls import petsc as petsc_nls
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 dot, div, dx, ds, dS, grad, inner, grad, avg, jump)

import commons, geometry, utils

warnings.simplefilter('ignore')


if __name__ == '__main__':
    encoding = io.XDMFFile.Encoding.HDF5
    adaptive_refine = False
    run_mesh = True
    micron = 1e-6
    markers = {
        'ne_pcc': 0,  # boundary not included yet
        'ne_se': 1,  # left boundary
        'pe_se': 2,  # internal boundary
        'pe_pcc': 3,  # left boundary
        'ne': 4,   # domain not included yet
        'se': 5,  # left domain
        'pe': 6,  # right domain
        "insulated_se": 7,  # top and bottom boundary - left
        "insulated_pam": 8,  # top and bottom boundary - right
    }
    mrkr = {
        "neg_cc_matrix": 1,
        "electrolyte": 2,
        # facets
        "left": 1,
        "right": 2,
        "middle": 3,
        "insulated": 4,
        "insulated_ncc": 5,
        "insulated_sep": 6,
    }

    workdir = "output/lithium-metal-leb"
    utils.make_dir_if_missing(workdir)
    output_meshfile = os.path.join(workdir, 'mesh.msh')
    tetr_meshfile = os.path.join(workdir, "tetr.xdmf")
    tria_meshfile = os.path.join(workdir, "tria.xdmf")
    line_meshfile = os.path.join(workdir, "line.xdmf")
    potential_resultsfile = os.path.join(workdir, "potential.bp")
    potential_dg_resultsfile = os.path.join(workdir, "potential_dg.bp")
    concentration_resultsfile = os.path.join(workdir, "concentration.bp")
    current_resultsfile = os.path.join(workdir, "current.bp")

    L_sep = 25 * micron
    L_neg_cc = 20 * micron
    L_sep_neg_cc = 15 * micron
    feature_radius = 5 * micron
    disk_radius = 100 * micron
    L_total = L_sep + L_neg_cc
    if run_mesh:
        gmsh.initialize()
        gmsh.model.add('lithium-metal-leb')

        # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5*micron)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", micron)

        neg_cc = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L_neg_cc, disk_radius)
        gmsh.model.occ.synchronize()
        sep_main = gmsh.model.occ.addCylinder(0, 0, L_neg_cc, 0, 0, L_sep, disk_radius)
        gmsh.model.occ.synchronize()
        sep_neg_cc = gmsh.model.occ.addCylinder(0, 0, L_neg_cc - L_sep_neg_cc, 0, 0, L_neg_cc, feature_radius)
        gmsh.model.occ.synchronize()
        current_collector = gmsh.model.occ.cut([(3, neg_cc)], [(3, sep_neg_cc)], removeTool=False)
        gmsh.model.occ.synchronize()
        electrolyte = gmsh.model.occ.fuse([(3, sep_main)], [(3, sep_neg_cc)])
        gmsh.model.occ.synchronize()
        
        
        volumes = gmsh.model.occ.getEntities(3)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [volumes[1][1]], mrkr["electrolyte"])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [volumes[0][1]], mrkr["neg_cc_matrix"])
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.occ.getEntities(2)
        left = []
        right = []
        middle = []
        insulated = []
        insulated_ncc = []
        insulated_sep = []
        for surf in surfaces:
            com = gmsh.model.occ.getCenterOfMass(surf[0], surf[1])
            if np.isclose(com[2], 0):
                left.append(surf[1])
            elif np.isclose(com[2], L_total):
                right.append(surf[1])
            elif np.isclose(com[2], L_total - 0.5 * L_sep) or np.isclose(com[2], 0.5 * L_neg_cc):
                # insulated.append(surf[1])
                if np.isclose(com[2], 0.5 * L_neg_cc):
                    insulated_ncc.append(surf[1])
                elif np.isclose(com[2], L_total - 0.5 * L_sep):
                    insulated_sep.append(surf[1])
            else:
                middle.append(surf[1])
        insulated = insulated_ncc + insulated_sep
        gmsh.model.addPhysicalGroup(2, left, mrkr["left"])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, right, mrkr["right"])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, middle, mrkr["middle"])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, insulated_ncc, mrkr["insulated_ncc"])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, insulated_sep, mrkr["insulated_sep"])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, insulated, mrkr["insulated"])
        gmsh.model.occ.synchronize()
        
        # gmsh.model.occ.dilate(gmsh.model.get_entities(1), 0, 0, 0, micron, micron, micron)
        gmsh.model.occ.synchronize()
        # adaptive refinement
        gmsh.model.mesh.generate(3)
        gmsh.write(output_meshfile)
        gmsh.finalize()
        
        mesh_3d = meshio.read(output_meshfile)
        tetr_mesh = geometry.create_mesh(mesh_3d, "tetra")
        # tetra_mesh = geometry.scale_mesh(tetr_mesh, "tetra", scale_factor=[micron, micron, micron])
        meshio.write(tetr_meshfile, tetr_mesh)
        tria_mesh = geometry.create_mesh(mesh_3d, "triangle")
        # tria_mesh = geometry.scale_mesh(tria_mesh, "triangle", scale_factor=[micron, micron, micron])
        meshio.write(tria_meshfile, tria_mesh)

    comm = MPI.COMM_WORLD
    full_mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(output_meshfile, comm, 0)

    # Create submesh for pe
    pam_domain, entity_map, vertex_map, geom_map = dolfinx.mesh.create_submesh(full_mesh, full_mesh.topology.dim, cell_tags.indices[(cell_tags.values == mrkr['electrolyte'])])

    # Transfer facet tags from parent mesh to submesh
    tdim = full_mesh.topology.dim
    fdim = tdim - 1
    c_to_f = full_mesh.topology.connectivity(tdim, fdim)
    f_map = full_mesh.topology.index_map(fdim)
    all_facets = f_map.size_local + f_map.num_ghosts
    all_values = np.zeros(all_facets, dtype=np.int32)
    all_values[facet_tags.indices] = facet_tags.values

    pam_domain.topology.create_entities(fdim)
    subf_map = pam_domain.topology.index_map(fdim)
    pam_domain.topology.create_connectivity(tdim, fdim)
    c_to_f_sub = pam_domain.topology.connectivity(tdim, fdim)
    num_sub_facets = subf_map.size_local + subf_map.num_ghosts
    sub_values = np.empty(num_sub_facets, dtype=np.int32)
    for i, entity in enumerate(entity_map):
        parent_facets = c_to_f.links(entity)
        child_facets = c_to_f_sub.links(i)
        for child, parent in zip(child_facets, parent_facets):
            sub_values[child] = all_values[parent]
    sub_meshtag = dolfinx.mesh.meshtags(pam_domain, pam_domain.topology.dim - 1, np.arange(
        num_sub_facets, dtype=np.int32), sub_values)
    pam_domain.topology.create_connectivity(pam_domain.topology.dim - 1, pam_domain.topology.dim)

    with dolfinx.io.XDMFFile(comm, "submesh.xdmf", "w", encoding=encoding) as xdmf:
        xdmf.write_mesh(pam_domain)
        xdmf.write_meshtags(sub_meshtag, x=pam_domain.geometry)

    # run simulation
    c_init = 30000  # mol/m3
    t = 0 # Start time
    eps = 1e-15
    dt = 1e-6
    T = 500 * dt

    dx = ufl.Measure("dx", domain=pam_domain)
    ds = ufl.Measure("ds", domain=pam_domain, subdomain_data=sub_meshtag)
    dS = ufl.Measure("dS", domain=pam_domain, subdomain_data=sub_meshtag)
    n = ufl.FacetNormal(pam_domain)
    tdim = pam_domain.topology.dim
    fdim = tdim - 1

    Q = fem.functionspace(pam_domain, ("CG", 2))
    c_n = fem.Function(Q)
    c_n.name = "c_n"
    c_n.interpolate(lambda x:  x[0] - x[0] + c_init)
    c_n.x.scatter_forward()

    # Create boundary condition
    # boundary_facets = sub_meshtag.find(mrkr['middle'])
    # bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(Q, fdim, boundary_facets), Q)

    ch = fem.Function(Q)
    ch.name = "concentration"
    ch.interpolate(lambda x: x[0] - x[0] + c_init)
    ch.x.scatter_forward()

    c = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)

    f = fem.Constant(pam_domain, PETSc.ScalarType(0))
    g = fem.Constant(pam_domain, PETSc.ScalarType(0))
    g_middle = fem.Constant(pam_domain, PETSc.ScalarType(1e-6/96485))
    D = fem.Constant(pam_domain, PETSc.ScalarType(1e-15))

    a = c * q * dx + dt * ufl.inner(D * ufl.grad(c), ufl.grad(q)) * dx
    L = (c_n + dt * f) * q * dx  + dt * ufl.inner(g, q) * ds(mrkr['insulated_sep']) + dt * ufl.inner(g, q) * ds(mrkr['right']) + dt * ufl.inner(g_middle, q) * ds(mrkr['middle'])

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    A = petsc.assemble_matrix(bilinear_form, bcs=[])
    A.assemble()
    b = fem.petsc.create_vector(linear_form)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    c_vtx = VTXWriter(comm, concentration_resultsfile, [ch], engine="BP4")
    c_vtx.write(0.0)
    count = 0
    while t < T:
        count += 1
        t += dt

        A = fem.petsc.assemble_matrix(fem.form(a), bcs=[])
        A.assemble()
        solver.setOperators(A)

        # Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(b, linear_form)

        # Apply Dirichlet boundary condition to the vector
        fem.petsc.apply_lifting(b, [bilinear_form], [[]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [])

        # Solve linear problem
        solver.solve(b, ch.vector)
        ch.x.scatter_forward()

        if count < 10:
            c_avg = fem.assemble_scalar(fem.form(ch * dx)) / fem.assemble_scalar(fem.form(1 * dx))
            print(f"average concentration: {c_avg}")

        # Update solution at previous time step (c_n)
        if np.any(ch.x.array < 0):
            print(f"Lithium depletion at {t:.2e} seconds")
            break
        c_n.x.array[:] = ch.x.array
        min_value = np.min(ch.x.array)
        c_vtx.write(t)
    c_vtx.close()

    # visualization
    bb_trees = bb_tree(pam_domain, pam_domain.topology.dim)
    n_points = 10000
    tol = 1e-8  # Avoid hitting the outside of the domain
    # x = np.linspace(100e-6 + tol, 200e-6 - tol, n_points)
    # y = np.ones(n_points) * 0.5 * 5 * 50e-6  # midline
    x = y = np.zeros(n_points)
    z = np.linspace(0 + tol, 45e-6 - tol, n_points)# midline
    points = np.zeros((3, n_points))
    points[0] = x
    points[1] = y
    points[2] = z
    u_values = []
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = compute_collisions_points(bb_trees, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = compute_colliding_cells(pam_domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = ch.eval(points_on_proc, cells)
    fig, ax = plt.subplots()
    ax.plot(points_on_proc[:, 2], u_values, "k", linewidth=2)
    ax.grid(True)
    ax.set_xlim([0, 45e-6])
    ax.set_ylim([0, c_init])
    ax.set_ylabel(r'$c$ [mol/m$^3$]', rotation=0, labelpad=50, fontsize='xx-large')
    ax.set_xlabel('[m]')
    ax.set_title('Concentration Across Midline')
    plt.tight_layout()
    plt.show()
