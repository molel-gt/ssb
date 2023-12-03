include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

exp: eapprox.o chkopts
	-${CLINKER} -o eapprox eapprox.o ${PETSC_LIB}
	${RM} eapprox.o

petscvec: petsc_vec.o chkopts
	-${CLINKER} -o petsc_vec petsc_vec.o ${PETSC_LIB}
	${RM} petsc_vec.o