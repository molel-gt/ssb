#include <petsc.h>

int main(int argc, char **argv)
{
    Mat A;
    Vec b, x;
    KSP ksp;
    int i;
    int j[4] = {0, 1, 2, 3};
    double ab[4] = {7.0, 1.0, 2.0, 4.0};
    double aA[4][4] = {
        {1, 0, 4, 2},
        {2, 6, 1, 5},
        {0, 1, -1, -2},
        {4, 3, -2, 1}
    };
    PetscInitialize(&argc, &argv, NULL, "ksp problem solution");
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, 4);
    VecSetFromOptions(b);
    VecSetValues(b, 4, j, ab, INSERT_VALUES);
    VecAssemblyBegin(b); VecAssemblyEnd(b);

    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 4, 4);
    MatSetFromOptions(A);
    MatSetUp(A);
    for (i = 0; i < 4; i++)
    {
        MatSetValues(A, 1, &i, 4, j, aA[i], INSERT_VALUES);

    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // FILE *fid;
    // fid = fopen("datafile.dat", "w");
    // double *abb;

    // VecGetArray(b, &abb);

    // for (int i=0; i <4; i++)
    // {
    //     fprintf(fid, "%d: %lf\n", i, abb[i]);

    // }
    // fclose(fid);
    // VecRestoreArray(b, &abb);
    VecDuplicate(b, &x);
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, b, x);
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    KSPDestroy(&ksp);
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);
    return PetscFinalize();
}