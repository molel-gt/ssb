#include <cmath>
#include "conduction.h"
#include <dolfinx.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <iostream>
#include <mpi.h>
#include <ufcx.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

using namespace dolfinx;
using namespace std;

namespace linalg
{
/// Compute vector r = alpha*x + y
/// @param[out] r Result
/// @param[in] alpha
/// @param[in] x
/// @param[in] y
template <typename U>
void axpy(la::Vector<U>& r, U alpha, const la::Vector<U>& x,
          const la::Vector<U>& y)
{
  std::transform(x.array().cbegin(), x.array().cend(), y.array().cbegin(),
                 r.mutable_array().begin(),
                 [alpha](auto x, auto y) { return alpha * x + y; });
}

/// Solve problem A.x = b using the Conjugate Gradient method
/// @tparam U The scalar type
/// @tparam ApplyFunction Type of the function object "action"
/// @param[in, out] x Solution vector, may be set to an initial guess
/// @param[in] b RHS Vector
/// @param[in] action Function that provides the action of the linear operator
/// @param[in] kmax Maximum number of iterations
/// @param[in] rtol Relative tolerances for convergence
/// @return The number if iterations
/// @pre It is required that the ghost values of `x` and `b` have been
/// updated before this function is called
template <typename U, typename ApplyFunction>
int cg(la::Vector<U>& x, const la::Vector<U>& b, ApplyFunction&& action,
       int kmax = 50, double rtol = 1e-8)
{
  // Create working vectors
  la::Vector<U> r(b), y(b);

  // Compute initial residual r0 = b - Ax0
  action(x, y);
  axpy(r, U(-1), y, b);

  // Create p work vector
  la::Vector<U> p(r);

  // Iterations of CG
  double rnorm0 = r.squared_norm();
  const double rtol2 = rtol * rtol;
  double rnorm = rnorm0;
  int k = 0;
  while (k < kmax)
  {
    ++k;

    // Compute y = A p
    action(p, y);

    // Compute alpha = r.r/p.y
    const U alpha = rnorm / la::inner_product(p, y);

    // Update x (x <- x + alpha*p)
    axpy(x, alpha, p, x);

    // Update r (r <- r - alpha*y)
    axpy(r, -alpha, y, r);

    // Update residual norm
    // Note: we use U for beta to support float, double, etc. U can be
    // complex, even though the value will always be real
    const double rnorm_new = r.squared_norm();
    const U beta = rnorm_new / rnorm;
    rnorm = rnorm_new;

    if (rnorm / rnorm0 < rtol2)
      break;

    // Update p (p <- beta*p + r)
    axpy(p, beta, p, r);
  }

  return k;
}
} // namespace linalg

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  {
    using T = PetscScalar;

    MPI_Comm comm = MPI_COMM_WORLD;

    // Create mesh and function space
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_box(
        comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {10, 10, 10}, mesh::CellType::tetrahedron,
        mesh::GhostMode::none));
    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_conduction_M, "ui", mesh));

    // Prepare and set Constants for the bilinear form
    auto f = std::make_shared<fem::Constant<T>>(0);
    auto g = std::make_shared<fem::Constant<T>>(0);

    // Define variational forms
    auto L = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_conduction_L, {V}, {}, {{"f", f}}, {}));

    // Action of the bilinear form "a" on a function ui
    auto ui = std::make_shared<fem::Function<T>>(V);
    auto M = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_conduction_M, {V}, {{"ui", ui}}, {{}}, {}));

    // Define boundary condition
    auto u_D = std::make_shared<fem::Function<T>>(V);
    u_D->interpolate(
        [](auto&& x) {
          return 0; //1 + xt::square(xt::row(x, 0)) + 2 * xt::square(xt::row(x, 1));
        });
    std::vector<std::int32_t> facets = mesh::exterior_facet_indices(*mesh);
    std::vector<std::int32_t> bdofs
        = fem::locate_dofs_topological({*V}, 1, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(u_D, bdofs);

    // Assemble RHS vector
    la::Vector<T> b(V->dofmap()->index_map, V->dofmap()->index_map_bs());
    fem::assemble_vector(b.mutable_array(), *L);

    // Apply lifting to account for Dirichlet boundary condition
    // b <- b - A * x_bc
    fem::set_bc(ui->x()->mutable_array(), {bc}, -1.0);
    dolfinx::fem::assemble_vector(b.mutable_array(), *M);

    // Communicate ghost values
    b.scatter_rev(common::IndexMap::Mode::add);

    // Set BC dofs to zero (effectively zeroes columns of A)
    fem::set_bc(b.mutable_array(), {bc}, 0.0);

    b.scatter_fwd();

    // Pack coefficients and constants
    auto coeff = fem::allocate_coefficient_storage(*M);
    const std::vector<T> constants = fem::pack_constants(*M);

    // Create function for computing the action of A on x (y = Ax)
    std::function<void(la::Vector<T>&, la::Vector<T>&)> action
        = [&M, &ui, &bc, &coeff, &constants](la::Vector<T>& x, la::Vector<T>& y)
    {
      // Zero y
      y.set(0.0);

      // Update coefficient ui (just copy data from x to ui)
      std::copy(x.array().begin(), x.array().end(),
                ui->x()->mutable_array().begin());

      // Compute action of A on x
      fem::pack_coefficients(*M, coeff);
      fem::assemble_vector(y.mutable_array(), *M, xtl::span<const T>(constants),
                           fem::make_coefficients_span(coeff));

      // Set BC dofs to zero (effectively zeroes rows of A)
      fem::set_bc(y.mutable_array(), {bc}, 0.0);

      // Accumuate ghost values
      y.scatter_rev(common::IndexMap::Mode::add);

      // Update ghost values
      y.scatter_fwd();
    };

    // Compute solution using the conjugate gradient method
    auto u = std::make_shared<fem::Function<T>>(V);
    int num_it = linalg::cg(*u->x(), b, action, 200, 1e-6);

    // Set BC values in the solution vectors
    fem::set_bc(u->x()->mutable_array(), {bc}, 1.0);

    // Compute L2 error (squared) of the solution vector e = (u - u_d, u
    // - u_d)*dx
    auto E = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_conduction_E, {}, {{"uexact", u_D}, {"usol", u}}, {}, {}, mesh));
    T error = fem::assemble_scalar(*E);

    if (dolfinx::MPI::rank(comm) == 0)
    {
      std::cout << "Number of CG iterations " << num_it << std::endl;
      std::cout << "Finite element error (L2 norm (squared)) "
                << std::abs(error) << std::endl;
    }
  }

  MPI_Finalize();

  return 0;
}


// using T = PetscScalar;

// int main(int argc, char **argv){
//     int size, rank;
//     // mesh::Mesh mesh;
//     dolfinx::init_logging(argc, argv);
//     PetscInitialize(&argc, &argv, nullptr, nullptr);
//     MPI_Init(&argc, &argv);
//     MPI_Comm comm = MPI_COMM_WORLD;
//     // io::XDMFFile file_sigma(mesh->comm(), "s51-51-51o0_0_0_tetr.xdmf", "r");
//     // file_sigma.read_mesh(*mesh);
//     int Nx = 51, Ny = 51, Nz = 51;
//     cout << "Trying out cxx dolfinx" << endl;
//     MPI_Finalize();
//     return 0;
// }