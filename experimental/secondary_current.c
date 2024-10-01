#include <petsc.h>

int main(int argc, char **argv)
{
    Mat A;
    Vec b, x;
    KSP ksp;
    int Nel = 10;
    /* remove rows corresponding to dirichlet bc,
        then add extra DoF due to discontinuity */
    int N = Nel - 1 + 1;
    double h = 1.0 / Nel; 
    int i;
    double ab[1] = {1.0/h};
    int j[1] = {N - 1};
    double gamma = 10;

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
        else if (i == 5){
            /* inner(grad(u), grad(v)) */
            num_entries = 2;
            int j[2] = {i-1, i};
            double entries[2] = {-1.0 / h, 1.0 / h};
            MatSetValues(A, 1, &i, num_entries, j, entries, INSERT_VALUES);
        }
        else if (i == 6){
            /* inner(grad(u), grad(v)) */
            num_entries = 2;
            int j[2] = {i, i+1};
            double entries[2] = {1.0 / h, -1.0 / h};
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
    fid = fopen("secondary_current.csv", "w");
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
