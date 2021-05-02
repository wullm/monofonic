CC = mpicc -qopenmp 
G99 = gcc -fopenmp

GSL_LIBS =  -lgsl -lgslcblas 

#CCFLAGS = $(CFLAGS) $(GSL_LIBS) -O3  -qopt-zmm-usage=high -vec-threshold0 -lfftw3  -lfftw3f   -lfftw3_omp -lfftw3_mpi -lfftw3f_mpi
CCFLAGS = $(CFLAGS) $(GSL_LIBS) -O3  -qopt-zmm-usage=high -vec-threshold0 -lfftw3  -lfftw3f   -lfftw3_omp -lfftw3_mpi -lfftw3f_mpi -DUSE_OPENMP 



pan_fftw3_test_code.x: main.o high_order_panphasia_routines.o  pan_mpi_routines.o uniform_rand_threefry4x64.o 
	$(CC) $(CCFLAGS) *.o  -o pan_fftw3_test_code.x
main.o:  main.c
	$(CC) $(CCFLAGS) -c main.c
high_order_panphasia_routines.o:  high_order_panphasia_routines.c
	$(CC) $(CCFLAGS) -c high_order_panphasia_routines.c
pan_mpi_routines.o:  pan_mpi_routines.c
	$(CC) $(CCFLAGS) -c pan_mpi_routines.c
uniform_rand_threefry4x64.o: uniform_rand_threefry4x64.c
	$(CC) $(CCFLAGS) -c uniform_rand_threefry4x64.c
clean:
	rm *.o
