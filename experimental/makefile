include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

exp: eapprox.o chkopts
	-${CLINKER} -o eapprox eapprox.o ${PETSC_LIB}
	${RM} eapprox.o

petscvec: petsc_vec.o chkopts
	-${CLINKER} -o petsc_vec petsc_vec.o ${PETSC_LIB}
	${RM} petsc_vec.o

ksp_prob: ksp_problem.o chkopts
	-${CLINKER} -o ksp_problem ksp_problem.o ${PETSC_LIB}
	${RM} ksp_problem.o

laplace_prob: laplace_ksp.o chkopts
	-${CLINKER} -o laplace_ksp laplace_ksp.o ${PETSC_LIB}
	${RM} laplace_ksp.o