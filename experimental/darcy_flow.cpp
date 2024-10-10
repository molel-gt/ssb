#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <iomanip>
#include <stdexcept>
#include <petscsnes.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

double dKk_dk(double k, gsl_mode_t precision){
    return 1/k * (1/(1 - pow(k, 2)) * gsl_sf_ellint_Ecomp(k, precision) - gsl_sf_ellint_Kcomp(k, precision));
}

double dFphik_dk(double k, double phi, gsl_mode_t precision){
    return 1/k * (1/(1 - pow(k, 2)) * gsl_sf_ellint_E(phi, k, precision) - gsl_sf_ellint_F(phi, k, precision));
}

double dEk_dk(double k, gsl_mode_t precision){
    return 1/k * (gsl_sf_ellint_Ecomp(k, precision) - gsl_sf_ellint_Kcomp(k, precision));
}

double dkprime_dk(double k){
    return k * pow(1 - pow(k, 2), -0.5);
}

// std::vector<double> F(double A, double a, double b, double L, double w, double h, gsl_mode_t precision){
//     std::vector<double> objective;
//     objective.resize(3);
//     double k = 1/b;
//     double phi = asin(a);
//     double kprime = sqrt(1 - pow(k, 2));
//     objective[0] = -0.5 * L + A * k * gsl_sf_ellint_Kcomp(k, precision);
//     objective[1] = -h + A * k * gsl_sf_ellint_Kcomp(kprime, precision);
    
//     objective[2] = w - 0.5 * L + A * k * gsl_sf_ellint_F(phi, k, precision);

//     return objective;

// }

// std::vector<std::vector<double>> J(double A, double a, double b, gsl_mode_t precision){
//     std::vector<std::vector<double>> jacobian;
//     double k = 1/b;
//     double phi = asin(a);
//     double kprime = sqrt(1 - pow(k, 2));
//     jacobian.push_back({ k * gsl_sf_ellint_Kcomp(k, precision), 0, -pow(k, 2) * gsl_sf_ellint_Kcomp(k, precision - pow(k, 3) * dKk_dk(k, precision))});
//     jacobian.push_back({ k * gsl_sf_ellint_Kcomp(kprime, precision), 0, dkprime_dk(k) * (-pow(k, 2) * gsl_sf_ellint_Kcomp(k, precision - pow(k, 3) * dKk_dk(k, precision)))});
//     jacobian.push_back({ k * gsl_sf_ellint_F(phi, k, precision), 0, 0});

//     return jacobian;
// }

// std::vector<double> get_parameters(double L, double w, double h, int max_iters, double tol){
//     std::vector<double> parameters;
//     parameters.resize(3);
//     double iters = 0;
//     double error = tol + 1;
//     while (iters < max_iters && error > tol){
//         // double new_parameters = parameters; - 
//         iters++;
//     }
//     return parameters;
// }

extern PetscErrorCode FormFunction(SNES, Vec, Mat, Mat, void *);
extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);
typedef struct {
    double L;
    double h;
    double w;
} AppCtx;

PetscErrorCode InitializeProblem(AppCtx *, double L, double h, double w);

int main(int argc, char** argv){
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("L", po::value<double>(), "L")
        ("s_w", po::value<double>(), "s/w")
        ("h_w", po::value<double>(), "h/w");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }
    // struct Parameters *parameters;
    AppCtx parameters;
    
    double L = vm["L"].as<double>();
    double w = L / (2 + vm["s_w"].as<double>());
    double h = w * vm["h_w"].as<double>();
    
    SNES        snes; /* nonlinear solver context */
    KSP         ksp;  /* linear solver context */
    PC          pc;   /* preconditioner context */
    Vec         x, r; /* solution, residual vectors */
    Mat         J;    /* Jacobian matrix */
    PetscMPIInt size;
    PetscScalar *xx;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, "Newton method to compute A, a, and b given L s/w and h/w."));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCall(InitializeProblem(&parameters, L, h, w));
    //PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Example is only for sequential runs");

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create nonlinear solver context
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
    PetscCall(SNESSetType(snes, SNESNEWTONLS));
    PetscCall(SNESSetOptionsPrefix(snes, "mysolver_"));
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix and vector data structures; set corresponding routines
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create vectors for solution and nonlinear function
  */
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, 3));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecDuplicate(x, &r));

  /*
     Create Jacobian matrix data structure
  */
    PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
    PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 3, 3));
    PetscCall(MatSetFromOptions(J));
    PetscCall(MatSetUp(J));
    PetscCall(SNESSetFunction(snes, r, FormFunction, (void *) &parameters));
    PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, (void *) &parameters));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Set linear solver defaults for this problem. By extracting the
     KSP and PC contexts from the SNES context, we can then
     directly call any KSP and PC routines to set various options.
  */
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetTolerances(ksp, 1.e-4, PETSC_CURRENT, PETSC_CURRENT, 20));

  /*
     Set SNES/KSP/KSP/PC runtime options, e.g.,
         -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
     These options will override those specified above as long as
     SNESSetFromOptions() is called _after_ any other customization
     routines.
  */
  PetscCall(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(VecGetArray(x, &xx));
    xx[0] = 2.0;
    xx[1] = 1.25;
    xx[2] = 1.5;
    PetscCall(VecRestoreArray(x, &xx));
  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */

  PetscCall(SNESSolve(snes, NULL, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&J));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode InitializeProblem(AppCtx *ctx, double L, double h, double w){
    PetscFunctionBeginUser;
    ctx->L = L;
    ctx->h = h;
    ctx->w = w;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, void *ctx){
    PetscFunctionBeginUser;

    gsl_mode_t precision = GSL_PREC_DOUBLE;
    const PetscScalar *xx;
    PetscScalar       *ff;
    double A = xx[0];
    double a = xx[1];
    double b = xx[2];
    AppCtx *parameters = (AppCtx *)ctx;
    double L = parameters->L;
    double w = parameters->w;
    double h = parameters->h;

    /* utility values */
    double k = 1/b;
    double phi = asin(a);
    double kprime = sqrt(1 - pow(k, 2));

    

    PetscCall(VecGetArrayRead(x, &xx));
    PetscCall(VecGetArray(f, &ff));
    /*
    Compute Function
    */
    ff[0] = -0.5 * L + A * k * gsl_sf_ellint_Kcomp(k, precision);
    ff[1] = -h + A * k * gsl_sf_ellint_Kcomp(kprime, precision);
    ff[2] = w - 0.5 * L + A * k * gsl_sf_ellint_F(phi, k, precision);

    /* Restore vectors */
    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArray(f, &ff));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormJacobian(SNES snes, Vec x, Mat jac, Mat B, void *ctx){
    gsl_mode_t precision = GSL_PREC_DOUBLE;
    const PetscScalar *xx;
    double A = xx[0];
    double a = xx[1];
    double b = xx[2];

    /* utility values */
    double k = 1/b;
    double phi = asin(a);
    double kprime = sqrt(1 - pow(k, 2));

    PetscScalar M[9];
    PetscInt idx[3] = {0, 1, 2};

    PetscFunctionBeginUser;

    PetscCall(VecGetArrayRead(x, &xx));
    /*
        Form Jacobian
    */

    M[0] = k * gsl_sf_ellint_Kcomp(k, precision);
    M[1] = 0;
    M[2] = -pow(k, 2) * gsl_sf_ellint_Kcomp(k, precision - pow(k, 3) * dKk_dk(k, precision));

    M[3] = k * gsl_sf_ellint_Kcomp(kprime, precision);
    M[4] = 0;
    M[5] = dkprime_dk(k) * (-pow(k, 2) * gsl_sf_ellint_Kcomp(k, precision - pow(k, 3) * dKk_dk(k, precision)));

    M[6] = k * gsl_sf_ellint_F(phi, k, precision);
    M[7] = A * k / sqrt((1 - pow(a, 2)) * (pow(b, 2) - pow(a, 2)));
    M[8] = -pow(k, 2) * gsl_sf_ellint_Kcomp(k, precision - pow(k, 3) * dFphik_dk(k, phi, precision));

    PetscCall(MatSetValues(B, 3, idx, 3, idx, M, INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(x, &xx));

    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    if (jac != B) {
        PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}
