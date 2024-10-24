include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
# include ${HOME}/lib/gsl
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

exp: eapprox.o chkopts
	-${CLINKER} -o eapprox eapprox.o ${PETSC_LIB}
	${RM} eapprox.o

petscvec: petsc_vec.o chkopts
	-${CLINKER} -o petsc_vec petsc_vec.o ${PETSC_LIB}
	${RM} petsc_vec.o

ksp_prob: ksp_problem.o
	-${CLINKER} -o ksp_problem ksp_problem.o ${PETSC_LIB}
	${RM} ksp_problem.o

laplace_prob: laplace_ksp.o chkopts
	-${CLINKER} -o laplace_ksp laplace_ksp.o ${PETSC_LIB}
	${RM} laplace_ksp.o

poisson_prob: poisson_ksp.o chkopts
	-${CLINKER} -o poisson_ksp laplace_ksp.o ${PETSC_LIB}
	${RM} poisson_ksp.o

secondary_current: secondary_current.o
	-${CLINKER} -o secondary_current secondary_current.o ${PETSC_LIB}
	${RM} secondary_current.o

tertiary_current: tertiary_current.o
	-${CLINKER} -o tertiary_current tertiary_current.o ${PETSC_LIB}
	${RM} tertiary_current.o

conduit_flow: conduit_flow.o
	-${CLINKER} -o conduit_flow conduit_flow.o -L${HOME}/opt/lib -lgsl -lblas -lm -lstdc++
	${RM} conduit_flow.o

# all: conduit_flow

# conduit_flow: conduit_flow.o
# 	gcc -g -o conduit_flow conduit_flow.o -lgsl -lm -lblas

# conduit_flow.o: conduit_flow.c
# 	gcc -c conduit_flow.c -L${HOME}/opt/lib

# clean:
# 	rm conduit_flow.o conduit_flow
