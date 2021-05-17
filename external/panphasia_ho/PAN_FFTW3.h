// Define macros for FFTW3 to allow swapping 
// between single/double precision FTs

#ifndef USE_PRECISION_FLOAT

#define FOURIER_DOUBLE

#endif


#ifdef FOURIER_DOUBLE
  #define FFTW_REAL double
  #define FFTW_PLAN                  fftw_plan
  #define FFTW_DESTROY_PLAN          fftw_destroy_plan
  #define FFTW_COMPLEX               fftw_complex
  #define FFTW_MALLOC                fftw_malloc
  #define FFTW_PLAN_DFT_1D           fftw_plan_dft_1d
  #define FFTW_PLAN_dft_3D           fftw_plan_dft_3d
  #define FFTW_EXECUTE               fftw_execute
  #define FFTW_DESTROY_PLAN          fftw_destroy_plan
  #define FFTW_FREE                  fftw_free
  #define FFTW_ALLOC_COMPLEX         fftw_alloc_complex
  #define FFTW_MPI_LOCAL_SIZE_MANY   fftw_mpi_local_size_many
  #define FFTW_MPI_LOCAL_SIZE_MANY_TRANSPOSED fftw_mpi_local_size_many_transposed
  #define FFTW_MPI_PLAN_MANY_TRANSPOSE  fftw_mpi_plan_many_transpose
  #define FFTW_MPI_EXECUTE_R2R       fftw_mpi_execute_r2r
  #define FFTW_PLAN_MANY_DFT         fftw_plan_many_dft
  #define FFTW_MPI_LOCAL_SIZE_3D     fftw_mpi_local_size_3d
  #define FFTW_MPI_PLAN_MANY_DTF     fftw_mpi_plan_many_dft
  #define FFTW_MPI_PLAN_MANY_DTF_R2C fftw_mpi_plan_many_dft_r2c
  #define FFTW_MPI_EXECUTE_DFT       fftw_mpi_execute_dft
  #define FFTW_MPI_EXECUTE_DFT_R2C   fftw_mpi_execute_dft_r2c
#else
  #define FFTW_REAL float
  #define FFTW_PLAN                  fftwf_plan
  #define FFTW_DESTROY_PLAN          fftwf_destroy_plan
  #define FFTW_COMPLEX               fftwf_complex
  #define FFTW_MALLOC                fftwf_malloc
  #define FFTW_PLAN_DFT_1D           fftwf_plan_dft_1d
  #define FFTW_PLAN_dft_3D           fftwf_plan_dft_3d
  #define FFTW_EXECUTE               fftwf_execute
  #define FFTW_DESTROY_PLAN          fftwf_destroy_plan
  #define FFTW_FREE                  fftwf_free
  #define FFTW_ALLOC_COMPLEX         fftwf_alloc_complex
  #define FFTW_MPI_LOCAL_SIZE_MANY   fftwf_mpi_local_size_many
  #define FFTW_MPI_LOCAL_SIZE_MANY_TRANSPOSED fftwf_mpi_local_size_many_transposed
  #define FFTW_MPI_PLAN_MANY_TRANSPOSE  fftwf_mpi_plan_many_transpose
  #define FFTW_MPI_EXECUTE_R2R       fftwf_mpi_execute_r2r
  #define FFTW_PLAN_MANY_DFT         fftwf_plan_many_dft
  #define FFTW_MPI_LOCAL_SIZE_3D     fftwf_mpi_local_size_3d
  #define FFTW_MPI_PLAN_MANY_DTF     fftwf_mpi_plan_many_dft
  #define FFTW_MPI_PLAN_MANY_DTF_R2C fftwf_mpi_plan_many_dft_r2c
  #define FFTW_MPI_EXECUTE_DFT       fftwf_mpi_execute_dft
  #define FFTW_MPI_EXECUTE_DFT_R2C   fftwf_mpi_execute_dft_r2c
#endif
