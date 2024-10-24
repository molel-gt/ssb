#include <petsc.h>

static char help[] = "Tertiary current distribution\n";

extern PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);

typedef struct {
    PetscReal R; // gas constant [J/K/mol]
    PetscReal T; // temperature [K]
    PetscReal F;  // Faraday constant [C/mol]
    PetscReal h;     /* mesh spacing */
    PetscMPIInt rank;
    PetscMPIInt size;
    // DM da; // distributed array
} AppCtx;

int main(int argc, char **argv)
{
    SNES snes;
    SNESLineSearch linesearch;
    PC pc;
    Mat A;
    Mat J; // Jacobian matrix
    Vec b, x, r;
    KSP ksp;
    AppCtx ctx;
    PetscInt N, Nel; // Number of nodes, Number of elements

    PetscFunctionBeginUser;
    PetscInitialize(&argc, &argv, NULL, help);
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &ctx.rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ctx.size));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &Nel, NULL));
    ctx.h = 1.0 / Nel;
    N = Nel; // discontinuity at midpoint with dirichlet bc at both ends
    PetscReal ab[1] = {1.0/h}; // because of dirichlet bc at right boundary
    // row and col indices
    PetscInt i;
    PetscInt j[1] = {N - 1};

    PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
    // PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, N, 1, 1, NULL, &ctx.da));
    // PetscCall(DMSetFromOptions(ctx.da));
    // PetscCall(DMSetUp(ctx.da))
    
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, N);
    VecSetFromOptions(b);
    VecZeroEntries(b);
    VecSetValues(b, 1, j, ab, INSERT_VALUES);
    VecAssemblyBegin(b); VecAssemblyEnd(b);

    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);
    MatSetFromOptions(A);
    MatSetUp(A);
    int num_entries = 3;
    for (i = 0; i < N; i++)
    {
        if (i == 0){
            num_entries = 2;
            int j[2] = {i, i+1};
            double entries[2] = {2.0 / h, -1.0 / h};
            MatSetValues(A, 1, &i, num_entries, j, entries, INSERT_VALUES);
        } else if (i == N - 1){
            num_entries = 2;
            int j[2] = {i-1, i};
            double entries[2] = {-1.0 / h, 2.0 / h};
            MatSetValues(A, 1, &i, num_entries, j, entries, INSERT_VALUES);
        }
        else {
            num_entries = 3;
            int j[3] = {i-1, i, i+1};
            double entries[3] = {-1.0 / h, 2.0 / h, -1.0 / h};
            MatSetValues(A, 1, &i, num_entries, j, entries, INSERT_VALUES);
        }
        

    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    VecDuplicate(b, &x);
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, b, x);
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    // write to file
    FILE *fid;
    fid = fopen("datafile.csv", "w");
    double *abb;

    VecGetArray(x, &abb);
    fprintf(fid, "x,u\n");
    fprintf(fid, "%lf,%f\n", 0.0, 0.0);

    for (int i=0; i < N; i++)
    {
        fprintf(fid, "%lf,%lf\n", (i+1)*h, abb[i]);

    }
    fprintf(fid, "%lf,%f\n", 1.0, 1.0);
    fclose(fid);
    VecRestoreArray(x, &abb);

    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);
    KSPDestroy(&ksp);
    return PetscFinalize();
}

PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, void *ctx){
    AppCtx *user = (AppCtx *)ctx;
    PetscFunctionBeginUser;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormJacobian(SNES snes, Vec x, Mat jac, Mat B, void *ctx){
    AppCtx *user = (AppCtx *)ctx;
    PetscFunctionBeginUser;

    PetscFunctionReturn(PETSC_SUCCESS);
}
