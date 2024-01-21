#include <gismo.h>
 
using namespace gismo;
 
int main(int argc, char *argv[])
{
    bool plot = false;
    index_t numRefine  = 5;
    index_t numElevate = 0;
    bool last = false;
    std::string fn("pde/poisson2d_bvp.xml");
 
    gsCmdLine cmd("Tutorial on solving a Poisson problem.");
    cmd.addInt( "e", "degreeElevation",
                "Number of degree elevation steps to perform before solving (0: equalize degree in all directions)", numElevate );
    cmd.addInt( "r", "uniformRefine", "Number of Uniform h-refinement loops",  numRefine );
    cmd.addString( "f", "file", "Input XML file", fn );
    cmd.addSwitch("last", "Solve solely for the last level of h-refinement", last);
    cmd.addSwitch("plot", "Create a ParaView visualization file with the solution", plot);
 
    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }
 
 
    gsFileData<> fd(fn);
    gsInfo << "Loaded file "<< fd.lastPath() << "\n";
 
    gsMultiPatch<> mp;
    fd.getId(0, mp);
 
    gsFunctionExpr<> f;
    fd.getId(1, f);
    gsInfo << "Source function "<< f << "\n";
 
    gsBoundaryConditions<> bc;
    fd.getId(2, bc);
    bc.setGeoMap(mp);
    gsInfo << "Boundary conditions:\n" << bc << "\n";
 
    gsFunctionExpr<> ms;
    fd.getId(3, ms);
 
    gsOptionList Aopt;
    fd.getId(4, Aopt);
 
 
    gsMultiBasis<> dbasis(mp, true);  // true: poly-splines (not NURBS)
    dbasis.setDegree( dbasis.maxCwiseDegree() + numElevate);
    if (last)
    {
        for (int r =0; r < numRefine; ++r)
            dbasis.uniformRefine();
        numRefine = 0;
    }
 
    gsInfo << "Patches: " << mp.nPatches() << ", degree: " << dbasis.minCwiseDegree() << "\n";
#ifdef _OPENMP
    gsInfo << "Available threads: " << omp_get_max_threads() << "\n";
#endif
 
    gsExprAssembler<> A(1,1);
    A.setOptions(Aopt);
 
    gsInfo << "Active options:\n" << A.options() << "\n";
 
    typedef gsExprAssembler<>::geometryMap geometryMap;
    typedef gsExprAssembler<>::variable    variable;
    typedef gsExprAssembler<>::space       space;
    typedef gsExprAssembler<>::solution    solution;
 
    // Elements used for numerical integration
    A.setIntegrationElements(dbasis);
    gsExprEvaluator<> ev(A);
 
    // Set the geometry map
    geometryMap G = A.getMap(mp);
 
    // Set the discretization space
    space u = A.getSpace(dbasis);
 
    // Set the source term
    auto ff = A.getCoeff(f, G);
 
    // Recover manufactured solution
    auto u_ex = ev.getVariable(ms, G);
 
    // Solution vector and solution variable
    gsMatrix<> solVector;
    solution u_sol = A.getSolution(u, solVector);
 
 
    gsSparseSolver<>::CGDiagonal solver;
 
    gsVector<> l2err(numRefine+1), h1err(numRefine+1);
    gsInfo << "(dot1=assembled, dot2=solved, dot3=got_error)\n"
        "\nDoFs: ";
    double setup_time(0), ma_time(0), slv_time(0), err_time(0);
    gsStopwatch timer;
    for (int r = 0; r <= numRefine; ++r)
    {
        gsInfo << "\nDebug print\n";
        dbasis.uniformRefine();
        gsInfo << "\nDebug print\n";
 
       // u.setup(bc, dirichlet::interpolation, 0);
        u.setup(bc, dirichlet::l2Projection, 0);
        gsInfo << "\nDebug print\n";
 
        // Initialize the system
        A.initSystem();
        setup_time += timer.stop();
        gsInfo << "\nDebug print\n";
 
        gsInfo << A.numDofs() << std::flush;
        gsInfo << "\nDebug print\n";
 
        timer.restart();
        A.assemble(
            igrad(u, G) * igrad(u, G).tr() * meas(G),
            u * ff * meas(G)
            );
 
        auto g_N = A.getBdrFunction(G);
        A.assembleBdr(bc.get("Neumann"), u * g_N.tr() * nv(G) );
 
        ma_time += timer.stop();
 
        gsInfo << "." << std::flush;
 
        timer.restart();
        solver.compute( A.matrix() );
        solVector = solver.solve(A.rhs());
        slv_time += timer.stop();
 
        gsInfo << "." << std::flush; // Linear solving done
 
        // omp_set_dynamic(0);     // Explicitly disable dynamic teams
        // omp_set_num_threads(1); // Use these threads for later parallel regions
 
        timer.restart();
        l2err[r]= math::sqrt( ev.integral( (u_ex - u_sol).sqNorm() * meas(G) ) );
        
        h1err[r]= l2err[r] +
            math::sqrt(ev.integral( ( igrad(u_ex) - igrad(u_sol,G) ).sqNorm() * meas(G) ));
        err_time += timer.stop();
        gsInfo << ". " << std::flush;
 
    }
 
 
    timer.stop();
    gsInfo << "\n\nTotal time: " << setup_time + ma_time + slv_time + err_time << "\n";
    gsInfo << "     Setup: " << setup_time << "\n";
    gsInfo << "  Assembly: " << ma_time    << "\n";
    gsInfo << "   Solving: " << slv_time   << "\n";
    gsInfo << "     Norms: " << err_time   << "\n";
 
    gsInfo << "\nL2 error: " << std::scientific << std::setprecision(3) << l2err.transpose() << "\n";
    gsInfo << "H1 error: " << std::scientific << h1err.transpose() << "\n";
 
    if (!last && numRefine>0)
    {
        gsInfo << "\nEoC (L2): " << std::fixed << std::setprecision(2)
              <<  ( l2err.head(numRefine).array()  /
                   l2err.tail(numRefine).array() ).log().transpose() / std::log(2.0)
                   << "\n";
 
        gsInfo <<   "EoC (H1): " << std::fixed << std::setprecision(2)
              <<( h1err.head(numRefine).array() /
                  h1err.tail(numRefine).array() ).log().transpose() / std::log(2.0) << "\n";
    }
 
    if (plot)
    {
        gsInfo << "Plotting in Paraview...\n";
 
        gsParaviewCollection collection("ParaviewOutput/solution", &ev);
        collection.options().setSwitch("plotElements", true);
        collection.options().setInt("plotElements.resolution", 16);
        collection.newTimeStep(&mp);
        collection.addField(u_sol,"numerical solution");
        collection.addField(u_ex, "exact solution");
        collection.saveTimeStep();
        collection.save();
 
 
        gsFileManager::open("ParaviewOutput/solution.pvd");
    }
    else
        gsInfo << "Done. No output created, re-run with --plot to get a ParaView "
                  "file containing the solution.\n";
 
    return EXIT_SUCCESS;
 
}
