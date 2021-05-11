#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <complex.h>
#include <fftw3-mpi.h>

#include "PAN_FFTW3.h"
#include "panphasia_functions.h"

extern size_t descriptor_base_size;

#ifdef USE_OPENMP
#include <omp.h>
int threads_ok;
int number_omp_threads = 1;
#endif

// does the same as the main below, but does not initialise MPI or FFTW (this should be done in MONOFONIC)
int PANPHASIA_HO_main(const char *descriptor, size_t *ngrid_load)
{
   int verbose = 0;
   int error;
   size_t x0 = 0, y0 = 0, z0 = 0;
   size_t rel_level;
   int fdim=1;   //Option to scale Fourier grid dimension relative to Panphasia coefficient grid

   //char descriptor[300] = "[Panph6,L20,(424060,82570,148256),S1,KK0,CH-999,Auriga_100_vol2]";

   PANPHASIA_init_descriptor_(descriptor, &verbose);

   printf("Descriptor %s\n ngrid_load %llu\n",descriptor,*ngrid_load);

   // Choose smallest value of level to equal of exceed *ngrid_load)

   for (rel_level=0; fdim*(descriptor_base_size<<(rel_level+1))<=*ngrid_load; rel_level++);

   printf("Setting relative level = %llu\n",rel_level);


   if (error = PANPHASIA_init_level_(&rel_level, &x0, &y0, &z0, &verbose))
   {
      printf("Abort: PANPHASIA_init_level_ :error code %d\n", error);
      abort();
   };

   //======================= FFTW ==============================

   ptrdiff_t alloc_local, local_n0, local_0_start;

   ptrdiff_t N0 = fdim*(descriptor_base_size << rel_level);

   alloc_local = FFTW_MPI_LOCAL_SIZE_3D(N0, N0, N0 +2 , MPI_COMM_WORLD, &local_n0, &local_0_start);

   FFTW_COMPLEX *Panphasia_White_Noise_Field;

   Panphasia_White_Noise_Field = FFTW_ALLOC_COMPLEX(alloc_local);

   if (error = PANPHASIA_compute_kspace_field_(rel_level, N0, local_n0, local_0_start, Panphasia_White_Noise_Field))
   {
      printf("Error code from PANPHASIA_compute ...  %d\n", error);
   };

   fftw_free(Panphasia_White_Noise_Field);

   return(0);
}

#ifdef STANDALONE_PANPHASIA_HO
int main(int argc, char **argv)
{

   int verbose = 0;
   int error;
   size_t x0 = 0, y0 = 0, z0 = 0;
   size_t rel_level;
   char descriptor[300] = "[Panph6,L20,(424060,82570,148256),S1,KK0,CH-999,Auriga_100_vol2]";

#ifdef USE_OPENMP
   omp_set_num_threads(number_omp_threads);
   int provided;
   MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
   threads_ok = provided >= MPI_THREAD_FUNNELED;
   if (threads_ok)
      threads_ok = fftw_init_threads();
   fftw_mpi_init();
   int num_threads = number_omp_threads;
   if (threads_ok)
   {
      fftw_plan_with_nthreads(num_threads);
   }
   else
   {
      printf("Failure to initialise threads ...\n");
      MPI_Finalize();
   };

   printf("OpenMP threads enabled with FFTW. Number of threads %d\n", fftw_planner_nthreads());
#else
   MPI_Init(&argc, &argv);
#endif

   PANPHASIA_init_descriptor_(descriptor, &verbose);

   rel_level = 6; //Set size of test dataset

   if (error = PANPHASIA_init_level_(&rel_level, &x0, &y0, &z0, &verbose))
   {
      printf("Abort: PANPHASIA_init_level_ :error code %d\n", error);
      abort();
   };

   //======================= FFTW ==============================

   fftw_mpi_init();

   ptrdiff_t alloc_local, local_n0, local_0_start;

   ptrdiff_t N0 = descriptor_base_size << rel_level;

   alloc_local = FFTW_MPI_LOCAL_SIZE_3D(N0, N0, N0+2, MPI_COMM_WORLD, &local_n0, &local_0_start);

   FFTW_COMPLEX *Panphasia_White_Noise_Field;

   Panphasia_White_Noise_Field = FFTW_ALLOC_COMPLEX(alloc_local);

   if (error = PANPHASIA_compute_kspace_field_(rel_level, N0, local_n0, local_0_start, Panphasia_White_Noise_Field))
   {
      printf("Error code from PANPHASIA_compute ...  %d\n", error);
   };

   fftw_free(Panphasia_White_Noise_Field);

   fftw_mpi_cleanup();
   //==================== End FFTW  ===========================

   MPI_Finalize();
   return(0);
}



#endif // STANDALONE_PANPHASIA_HO
