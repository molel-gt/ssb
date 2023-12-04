/*   DMDA/KSP solving a system of linear equations.
     Poisson equation in 2D:

     div(grad u) = 0,  0 < x,y < 1
     with
       forcing function f = 0,
       Neumann boundary conditions
        dp/dx = 0 for y = 0, x = 1.
       Dirichlet boundary conditions
        u = 0 for y = 0
        u = 1 for y = 1.

     Contributed by Leshinka Molel <emolel3@gatech.edu>, 2023,
         based on petsc/src/ksp/ksp/tutorials/ex50.c

     Example of Usage:
          ./poisson_ksp -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 3 -ksp_monitor -ksp_view -dm_view draw -draw_pause -1
          ./poisson_ksp -da_grid_x 100 -da_grid_y 100 -pc_type mg  -pc_mg_levels 1 -mg_levels_0_pc_type ilu -mg_levels_0_pc_factor_levels 1 -ksp_monitor -ksp_view
          ./poisson_ksp -da_grid_x 100 -da_grid_y 100 -pc_type mg -pc_mg_levels 1 -mg_levels_0_pc_type lu -mg_levels_0_pc_factor_shift_type NONZERO -ksp_monitor
          mpiexec -n 4 ./poisson_ksp -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 10 -ksp_monitor -ksp_view -log_view
*/

#include <petsc.h>


extern PetscErrorCode formRHS(KSP, Vec, void *);
extern PetscErrorCode formJacobian(KSP, Mat, Mat, void *);

typedef struct {
  PetscScalar uu, tt;
} UserContext;


int main(int argc, char **argv)
{
    DM da;
    KSP ksp;
    UserContext user;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, "Solve 2D Poisson"));
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 9, 9,
    PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));

    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(KSPSetDM(ksp, (DM)da));
    PetscCall(DMSetApplicationContext(da, &user));
    user.uu = 1.0;
    user.tt = 1.0;

    PetscCall(KSPSetComputeRHS(ksp, formRHS, &user));
    PetscCall(KSPSetComputeOperators(ksp, formJacobian, &user));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, NULL, NULL));

    PetscCall(DMDestroy(&da));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(PetscFinalize());
    return 0;
}


PetscErrorCode formRHS(KSP ksp, Vec b, void *ctx)
{
    DMDALocalInfo info;
    PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode formJacobian(KSP ksp, Mat J, Mat jac, void *ctx)
{
    PetscInt     i, j, M, N, xm, ym, xs, ys, num, numi, numj;
    PetscScalar  v[5], Hx, Hy, HydHx, HxdHy;
    MatStencil   row, col[5];
    DM           da;
    MatNullSpace nullspace;

    PetscFunctionBeginUser;
    PetscCall(KSPGetDM(ksp, &da));
    PetscCall(DMDAGetInfo(da, 0, &M, &N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

    Hx = 1.0 / (PetscReal)(M);
    Hy = 1.0 / (PetscReal)(N);
    HxdHy = Hx / Hy;
    HydHx = Hy / Hx;

    PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));
    for (j = ys; j < ys + ym; j++)
    {
        for (i = xs; i < xs + xm; i++)
        {
            row.i = i;
            row.j = j;

            if (i == 0 || j == 0 || i == M - 1 || j == N - 1) // at grid boundaries
            {
                num = 0;
                numi = 0;
                numj = 0;
                if (j != 0)
                {
                    v[num] = -HxdHy;
                    col[num].i = i;
                    col[num].j = j - 1;
                    num++;
                    numj++;
                }
                if (i != 0)
                {
                    v[num] = -HydHx;
                    col[num].i = i - 1;
                    col[num].j = j;
                    num++;
                    numi++;
                }
                if (i != M - 1)
                {
                    v[num] = -HydHx;
                    col[num].i = i + 1;
                    col[num].j = j;
                    num++;
                    numi++;
                }
                if (j != N - 1)
                {
                    v[num] = -HxdHy;
                    col[num].i = i;
                    col[num].j = j + 1;
                    num++;
                    numj++;
                }
                v[num] = ((PetscReal)(numj) * HxdHy + (PetscReal)(numi) * HydHx);
                col[num].i = i;
                col[num].j = j;
                num++;
                PetscCall(MatSetValuesStencil(jac, 1, &row, num, col, v, INSERT_VALUES));
            }
            else
            {
                v[0] = -HxdHy;
                col[0].i = i;
                col[0].j = j - 1;
                v[1] = -HydHx;
                col[1].i = i - 1;
                col[1].j = j;
                v[2] = 2.0 * (HxdHy + HydHx);
                col[2].i = i;
                col[2].j = j;
                v[3] = -HydHx;
                col[3].i = i + 1;
                col[3].j = j;
                v[4] = -HxdHy;
                col[4].i = i;
                col[4].j = j + 1;
                PetscCall(MatSetValuesStencil(jac, 1, &row, num, col, v, INSERT_VALUES));
            }
        }
    }
    PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));

    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace));
    PetscCall(MatSetNullSpace(J, nullspace));
    PetscCall(MatNullSpaceDestroy(&nullspace));
    PetscFunctionReturn(PETSC_SUCCESS);
}