#include <petsc.h>

int main(int argc, char **argv){
    Mat A;
    Vec b;
    int i;
    int j[4] = {0, 1, 2, 3};
    double ab[4] = {7.0, 1.0, 2.0, 4.0};
    double aA[4][4] = {
        {1, 0, 4, 2},
        {2, 6, 1, 5},
        {0, 1, -1, -2},
        {4, 3, -2, 1}
    };
    PetscInitialize(&argc, &argv, NULL, "Vector manipulation");
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

    VecDestroy(&b);
    MatDestroy(&A);
    return PetscFinalize();
}