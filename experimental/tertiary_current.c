#include <petsc.h>

static char help[] = "Tertiary current distribution\n";

extern PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);

typedef struct {
    // PetscReal R; // gas constant [J/K/mol]
    // PetscReal T; // temperature [K]
    // PetscReal F;  // Faraday constant [C/mol]
    PetscReal h;     /* mesh spacing */
    // PetscMPIInt rank;
    PetscMPIInt size;
    PetscMPIInt N;
    // DM da; // distributed array
    PetscReal gamma;
} AppCtx;

int main(int argc, char **argv)
{
    SNES snes;
    PC pc;
    Mat J; // Jacobian matrix
    Vec b, x, r;
    KSP ksp;
    AppCtx ctx;
    PetscInt N; // Number of nodes/elements
    PetscScalar *xx;
    MPI_Comm comm;

    PetscFunctionBeginUser;
    PetscInitialize(&argc, &argv, NULL, help);
    // PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &ctx.rank));
    comm = PETSC_COMM_WORLD;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ctx.size));
    PetscCheck(ctx.size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "Example is only for sequential runs");
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &ctx.N, NULL));
    
    N = ctx.N; // discontinuity at midpoint with dirichlet bc at both ends
    ctx.h = 1.0 / (N - 1);
    ctx.gamma = 25.0;

    PetscCall(SNESCreate(comm, &snes));
    PetscCall(SNESSetType(snes, SNESNEWTONLS));
    PetscCall(SNESSetOptionsPrefix(snes, "mysolver_"));

    /*create required vectors and matrices*/
    PetscCall(VecCreate(comm, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, ctx.N));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecDuplicate(x, &r));
    /* create rhs */
    double ab[1] = {1.0/ctx.h};
    int j[1] = {N - 1};
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, N);
    VecSetFromOptions(b);
    VecZeroEntries(b);
    VecSetValues(b, 1, j, ab, INSERT_VALUES);
    VecAssemblyBegin(b); VecAssemblyEnd(b);

    /* create jacobian matrix structure */

    PetscCall(MatCreate(comm, &J));
    PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, ctx.N, ctx.N));
    PetscCall(MatSetFromOptions(J));
    PetscCall(MatSetUp(J));

    /* set utility functions */
    PetscCall(SNESSetFunction(snes, r, FormFunction, &ctx));
    PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, &ctx));

    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(PCSetType(pc, PCLU));
    PetscCall(KSPSetTolerances(ksp, 1.e-8, 1e-6, PETSC_CURRENT, 50));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(SNESSetFromOptions(snes));


    /* initial guess */
    PetscCall(VecGetArray(x, &xx));
    for (int i=0; i<ctx.N; i++){
        xx[i] = 1.0;//i * ctx.h;
    }
    /* solve */
    PetscCall(SNESSolve(snes, b, x));

    PetscCall(VecRestoreArray(x, &xx));

    // write to file
    FILE *fid;
    fid = fopen("datafile.csv", "w");
    double *abb;

    VecGetArray(x, &abb);
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    fprintf(fid, "x,u\n");
    fprintf(fid, "%lf,%f\n", 0.0, 0.0);

    for (int i=0; i < N; i++)
    {
        if(i < 5) {
            fprintf(fid, "%lf,%lf\n", (i+1)*ctx.h, abb[i]);
        }
        else {
            fprintf(fid, "%lf,%lf\n", (i)*ctx.h, abb[i]);
        }

    }
    fprintf(fid, "%lf,%f\n", 1.0, 1.0);
    fclose(fid);
    VecRestoreArray(x, &abb);


    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&r));
    PetscCall(MatDestroy(&J));
    PetscCall(SNESDestroy(&snes));
    return PetscFinalize();
}

PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, void *ctx){
    AppCtx *user = (AppCtx *)ctx;
    const PetscScalar *xx;
    PetscScalar *ff;
    PetscReal gamma = user->gamma;
    PetscReal h = user->h;
    PetscInt N = user->N;
    PetscReal nl = 1.0;
    PetscReal nr = -1.0;

    PetscFunctionBeginUser;
    PetscCall(VecGetArrayRead(x, &xx));
    PetscCall(VecGetArray(f, &ff));

    for (int i = 0; i < N; i++){
        if (i == 0){
            ff[i] = 2.0/h * xx[i] - 1.0/h*xx[i+1];
        }
        else if((i > 0 && i < 4) || (i > 5 && i < N-1)){
            ff[i] = -1.0/h * xx[i-1] + 2.0/h * xx[i] -1.0 * xx[i+1];
        }
        else if(i == 4){// left of discontinuity
            // ff[i] = 1.0/h*xx[i] - 0.5 * 1.0/h * (xx[i] - xx[i+1])*nl - 0.5 * 1.0/h * (xx[i+1] - xx[i])*nl + 1.0 * gamma/h*1.0/h * (xx[i+1] - xx[i])*nl;
            ff[i] = -1.0/h*xx[i-1] + 1.0/h*xx[i] + 1.0 * gamma/h * 1.0/h * (xx[i+1] - xx[i]) * nl;
        }
        else if(i == 5){// right of discontinuity
            // ff[i] = 1.0/h*xx[i] + 0.5 * 1.0/h * (xx[i-1] + xx[i])*nr + 0.5 * 1.0/h * (xx[i] - xx[i-1])*nr - 1.0 * gamma/h * 1.0/h * (xx[i] - xx[i-1])*nr;
            ff[i] = -1.0/h*xx[i+1] + 1.0/h*xx[i] - 1.0 * gamma/h * 1.0/h * (xx[i] - xx[i-1])*nr;
        }
        else{
            ff[i] = -1.0/h * xx[i-1] + 2.0/h * xx[i];
        }
    }

    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArray(f, &ff));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormJacobian(SNES snes, Vec x, Mat jac, Mat B, void *ctx){
    AppCtx *user = (AppCtx *)ctx;
    const PetscScalar *xx;
    PetscReal gamma = user->gamma;
    PetscReal h = user->h;
    PetscReal nl = 1.0;
    PetscReal nr = -1.0;
    PetscInt N = user->N;

    PetscFunctionBeginUser;
    PetscCall(VecGetArrayRead(x, &xx));
    for (int i = 0; i < N; i++){
        PetscInt rows[1] = {i};
        if (i == 0){
            PetscInt cols[2] = {i, i+1}; 
            PetscScalar A[2] = {2.0/h, -1.0/h};
            PetscCall(MatSetValues(B, 1, rows, 2, cols, A, INSERT_VALUES));
        }
        else if((i > 0 && i < 4) || (i > 5 && i < N - 1)){
            PetscInt cols[3] = {i-1, i, i+1};
            PetscScalar A[3] = {1.0/h, 2.0/h, -1.0/h};
            PetscCall(MatSetValues(B, 1, rows, 3, cols, A, INSERT_VALUES));
        }
        else if(i == 4){// left of discontinuity
            PetscInt cols[3] = {i-1, i, i+1};
            // PetscScalar A[2] = {1.0/h * (1.0 - gamma/h * nl), 1.0/h * (1.0 + gamma/h) * nl};
            PetscScalar A[3] = {-1.0/h, 1.0/h*(1.0 - gamma/h * nl), 1.0/h*gamma/h*nl};
            PetscCall(MatSetValues(B, 1, rows, 3, cols, A, INSERT_VALUES));
        }
        else if(i == 5){// right of discontinuity
            PetscInt cols[3] = {i-1, i, i+1};
            // PetscScalar A[2] = {1.0/h * (1.0 + gamma/h) * nr, 1.0/h * (1.0 - gamma/h * nr)};
            PetscScalar A[3] = {1.0/h*gamma/h*nr, 1.0/h*(1.0 - gamma/h * nr), -1.0/h};
            PetscCall(MatSetValues(B, 1, rows, 3, cols, A, INSERT_VALUES));
        }
        else{
            PetscInt cols[2] = {i-1, i};
            PetscScalar A[2] = {-1.0/h, 2.0/h};
            PetscCall(MatSetValues(B, 1, rows, 2, cols, A, INSERT_VALUES));
        }
    }

    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    if (jac != B) {
        PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  }
    PetscFunctionReturn(PETSC_SUCCESS);
}
