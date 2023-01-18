
.PHONY: compile clean runsyrk rungemm runsymv

FLAGSMKL=-fsycl -DMKL_ILP64  -I"${MKLROOT}/include"
LINKMKL=-L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl



compile: program_syrk.x program_gemm.x program_symv.x

clean:
	rm -f *.x

runsyrk: program_syrk.x
	while true; do ./$< || break; done

rungemm: program_gemm.x
	while true; do ./$< || break; done

runsymv: program_symv.x
	while true; do ./$< || break; done



program_syrk.x: source_syrk.cpp Makefile
	icpx -std=c++17 -g -O3 -fopenmp ${FLAGSMKL} $< -o $@ ${LINKMKL}

program_gemm.x: source_gemm.cpp Makefile
	icpx -std=c++17 -g -O3 -fopenmp ${FLAGSMKL} $< -o $@ ${LINKMKL}

program_symv.x: source_symv.cpp Makefile
	icpx -std=c++17 -g -O3 -fopenmp ${FLAGSMKL} $< -o $@ ${LINKMKL}
