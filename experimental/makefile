include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

eappox: eapprox.o chkopts
	-${CLINKER} -o eapprox eapprox.o ${PETSC_LIB}
	${RM} eapprox.o