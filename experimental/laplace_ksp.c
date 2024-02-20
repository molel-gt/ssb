#include <petsc.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    Vec b, x;
    Mat A;
    KSP ksp;

    int m;
    int startind, endind;
    int k[1] = {0};
    double ab[1] = {1.0};
    m = atoi(argv[1]);

    PetscInitialize(&argc, &argv, NULL, "Solve Laplacian\n");

    // create vectors
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, m);
    VecSetFromOptions(b);
    VecDuplicate(b, &x);

    VecSetValues(b, 1, k, ab, INSERT_VALUES);
    VecAssemblyBegin(b); VecAssemblyEnd(b);

    // create matrix
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, m);
    MatSetFromOptions(A);
    MatSetUp(A);

    MatGetOwnershipRange(A, &startind, &endind);
    int i;
    int j[3];
    double v[3];
    for (i = startind; i < endind; i++)
    {
        if (i == 0){
            v[0] = 2; v[1] = -1;
            j[0] = 0; j[1] = 1;
            MatSetValues(A, 1, &i, 2, j, v, INSERT_VALUES);
        }
        else if (i == m - 1)
        {
            v[m-1] = -1;
            j[0] = m-1;
            MatSetValues(A, 1, &i, 1, j, v, INSERT_VALUES);

        }
        else
        {
            v[0] = -1; v[1] = 2; v[2] = -1;
            j[0] = i-1; j[1] = i; j[2] = i+1;
            MatSetValues(A, 1, &i, 3, j, v, INSERT_VALUES);

        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    KSPCreate(PETSC_COMM_WORLD, &ksp);
    
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, b, x);

    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);
    KSPDestroy(&ksp);
    return PetscFinalize();
}