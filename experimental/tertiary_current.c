#include <petsc.h>
#include <slepc.h>

int main(int argc, char **argv)
{
    Mat A;
    Vec b, x;
    KSP ksp;
    int Nel = 10;
    int N = Nel - 1; // remove rows corresponding to dirichlet bc
    double h = 1.0 / Nel; 
    int i;
    double ab[1] = {1.0/h};
    int j[1] = {N - 1};

    PetscInitialize(&argc, &argv, NULL, "ksp problem solution");
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

    // calculation of condition number
    PetscInt nconv, ii;
    EPS eps;  // eigensolver context
    Vec xr, xi; // eigen vectors
    PetscScalar kr, ki; // eigenvalues
    PetscCall(SlepcInitialize(&argc,&argv,(char*)0,NULL));
    PetscCall(MatCreateVecs(A, NULL, &xr));
    PetscCall(MatCreateVecs(A, NULL, &xi));
    PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
    PetscCall(EPSSetOperators(eps, A, NULL));
    PetscCall(EPSSetProblemType(eps, EPS_HEP));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetConverged(eps, &nconv));
    for (ii = 0; ii < nconv; ii++){
        PetscCall(EPSGetEigenpair(eps, ii, &kr, &ki, xr, xi));
    }
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "           k          ||Ax-kx||/||kx||\n"
    //     "   ----------------- ------------------\n"));
    EPSDestroy(&eps);

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
