#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <complex.h>
#include <fftw3-mpi.h>

#include "PAN_FFTW3.h"
#include "panphasia_functions.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif


extern const int Nbasis;
extern const int irank_p[3][84];

extern size_t descriptor_order;
extern size_t descriptor_kk_limit;


int PANPHASIA_compute_kspace_field_(size_t relative_level, ptrdiff_t N0_grid,
				     ptrdiff_t local_n0_return, ptrdiff_t local_0_start_return,
                                     FFTW_COMPLEX *return_field)
{

size_t copy_list[Nbasis];
int fdim=1;
int pmax = 6;
size_t ncopy = (pmax+1)*(pmax+2)*(pmax+3)/6;
size_t xorigin=local_0_start_return, yorigin=0, zorigin=0;
size_t xextent =local_n0_return, yextent = N0_grid, zextent = N0_grid;
int verbose = 1;
int flag_output_mode=2;
int error;
ptrdiff_t size_to_alloc;
FFTW_PLAN output_coeff_forward_plan; 


if (pmax>descriptor_order) return(100000);

for (size_t i=0; i<Nbasis; i++) copy_list[i]=i;

printf("Dimensions of FT (%td,%td,%td)\n",N0_grid,N0_grid,N0_grid);
printf("local_no %td local_0_start %td\n",local_n0_return, local_0_start_return);



// Distribution for ncopy 3-D arrays //
 {
int rank =3;
 const ptrdiff_t ndimens_alloc[] = {N0_grid, N0_grid, N0_grid+2}; // Allocated for r2c
ptrdiff_t howmany = ncopy;
ptrdiff_t local_n0;
ptrdiff_t local_0_start;

size_to_alloc = FFTW_MPI_LOCAL_SIZE_MANY(rank, ndimens_alloc, howmany,
 						    FFTW_MPI_DEFAULT_BLOCK,MPI_COMM_WORLD,
                                                    &local_n0,&local_0_start);
printf("size_to_alloc = %td\n",size_to_alloc);
printf("cf value %ld\n",ncopy*xextent*yextent*zextent);
printf("local_n0 %td local_0_start %td\n",local_n0,local_0_start);

 };

 void *output_coefficients= FFTW_MALLOC(sizeof(FFTW_REAL)*size_to_alloc);

 if (output_coefficients==NULL) return(100001);

 FFTW_REAL     *ptr_real_output_coefficients = output_coefficients;
 FFTW_COMPLEX *ptr_cmplx_output_coefficients = output_coefficients;


 printf("Making the plan ... \n");

//////////////////// Make plan for ncopy interleaved FTs ///////////////////////////

{
  int rank = 3;
  const ptrdiff_t ndimens[3] =  {N0_grid, N0_grid, N0_grid};
  ptrdiff_t howmany = ncopy;
  ptrdiff_t block = FFTW_MPI_DEFAULT_BLOCK;
  ptrdiff_t tblock =  FFTW_MPI_DEFAULT_BLOCK;
  unsigned flags = FFTW_ESTIMATE;

   output_coeff_forward_plan = FFTW_MPI_PLAN_MANY_DTF_R2C(rank, ndimens,
                                 howmany, block, tblock,
                                 ptr_real_output_coefficients, ptr_cmplx_output_coefficients,
                                 MPI_COMM_WORLD, flags);
   if (output_coeff_forward_plan==NULL) {
   printf("Null plan\n");
   return(100051);
   };

 };
//////////////////////////////////////////////////////////////////////////   
   



printf("Plan completed ... \n");

 printf("xorigin,yorigin,zorigin (%ld,%ld,%ld)\n ",xorigin,yorigin,zorigin);
 printf("xextent,yextent,zextent (%ld,%ld,%ld)\n ",xextent,yextent,zextent);


 
if (error = PANPHASIA_compute_coefficients_(&xorigin,&yorigin,&zorigin,&xextent,&yextent,
					 &zextent, copy_list, &ncopy,
					    ptr_real_output_coefficients,&flag_output_mode,&verbose)){
return(100100+error);
 };

 for (int j=0; j<4; j++){
   for (int i=0; i<4; i++) printf("(%lf ) ",ptr_real_output_coefficients[j+ i*ncopy]);
				printf("\n");
 };

{

 size_t nfft_dim; 
nfft_dim = N0_grid;
int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  char filename[100];
  sprintf(filename,"output_real_space_field.%d",rank);
  
  FILE *fp;

  fp = fopen(filename,"w");

  for (int ix=0; ix<local_n0_return; ix++)
    for (int iy=0; iy < nfft_dim; iy++)
      for (int iz=0; iz < nfft_dim; iz++){
	int index = ix*N0_grid*(N0_grid+2) + iy*(N0_grid+2) + iz;
        fprintf(fp,"%6d%6d%6d %14.8lf %d\n",ix+local_0_start_return,iy,iz,
		ptr_real_output_coefficients[index],index);
		      };

 fclose(fp);
 };


 

FFTW_MPI_EXECUTE_DFT_R2C(output_coeff_forward_plan,ptr_real_output_coefficients,ptr_cmplx_output_coefficients);


for (int j=0; j<4; j++){
   for (int i=0; i<4; i++) printf("(%lf  %lf) ",creal(ptr_cmplx_output_coefficients[j+ i*ncopy]),
				cimag(ptr_cmplx_output_coefficients[j + i*ncopy])); printf("\n");
 };

 


//  Compute 1-D Spherical Bessel coefficients for each order.
 size_t nfft_dim; 
  nfft_dim = N0_grid;
 if (nfft_dim<N0_grid) nfft_dim=N0_grid; if (nfft_dim<N0_grid) nfft_dim=N0_grid; 
 size_t  n4dimen;
 n4dimen=(nfft_dim%4==0) ? 4*(nfft_dim/4)+4 : 4*(nfft_dim/4)+5; 
	  double complex *sph_bessel_coeff = FFTW_MALLOC(sizeof(double complex)*n4dimen*(pmax+1)); 

	  compute_sph_bessel_coeffs(nfft_dim, pmax, n4dimen, fdim, sph_bessel_coeff);

	  printf("Reached here! ndimen4 %ld\n",n4dimen);

 


{
size_t index1,index2;
complex weight;
size_t    p_total = (pmax+1)*(pmax+2)*(pmax+3)/6;
int m;
 memset(return_field,0, local_n0_return*N0_grid*N0_grid * sizeof(FFTW_COMPLEX));

#ifdef USE_OPENMP
#pragma omp parallel for collapse(3)  \
  private (index1,index2,weight,m)
#endif
  for(int ix=0;ix<local_n0_return;ix++)
     for(int iy=0;iy<nfft_dim;iy++)
        for(int iz=0;iz<=nfft_dim/2;iz++){
	  index1 = ix*N0_grid*(N0_grid/2+1) + iy*(N0_grid/2+1) + iz;
        for (int m=0; m<p_total; m++){
        index2 = p_total*index1 + m;
        weight = sph_bessel_coeff[n4dimen*irank_p[0][m]+ix+local_0_start_return]*
	      sph_bessel_coeff[n4dimen*irank_p[1][m]+iy]*
	                                sph_bessel_coeff[n4dimen*irank_p[2][m]+iz];
        return_field[index1] += weight * ptr_cmplx_output_coefficients[index2];
        };
	};
 };
 
 
 printf("Reached here 10!\n");

   //Add phase shift and normalise field
{

   double complex phase_shift_and_scale;
   int  kx,ky,kz;
   const double pi = 4.0 * atan(1.0);
   size_t index1;

#ifdef USE_OPENMP
#pragma omp parallel for collapse(3)  \
  private (index1,kx,ky,kz,phase_shift_and_scale)
#endif
  for(int ix=0;ix<local_n0_return;ix++)
    for(int iy=0;iy<nfft_dim;iy++)
      for(int iz=0;iz<=nfft_dim/2;iz++){
	index1 = ix*N0_grid*(N0_grid/2+1) + iy*(N0_grid/2+1) + iz;
        kx = (ix+local_0_start_return>nfft_dim/2) ? 
              ix + local_0_start_return - nfft_dim : ix + local_0_start_return;
        ky = (iy > nfft_dim/2) ? iy-nfft_dim : iy;
        kz = (iz > nfft_dim/2) ? iz-nfft_dim : iz;

        if ( (kx==nfft_dim/2)||(ky==nfft_dim/2)||(kz==nfft_dim/2)){
	  // Set Nyquist modes to zero - not used by IC_Gen anyway.
          phase_shift_and_scale = 0.0; //1.0/pow((double)nfft_dim,1.5);  // No phase shift
        }else{
        phase_shift_and_scale = 
	  cexp( (-I)*pi*(double)(kx + ky + kz)/(double)nfft_dim)/pow((double)nfft_dim,1.5);
        };

	return_field[index1] *= phase_shift_and_scale;
     
      };


 };

 printf("Reached here 11!\n");



 // Rescale selected Fourier modes to unit amplitude.
 // By default this part is not executed. 

 if (descriptor_kk_limit>0){
   size_t index1;
   complex weight;
   size_t ksquared;
   int kx,ky,kz;
#ifdef USE_OPENMP
#pragma omp parallel for collapse(3)  \
  private (index1,kx,ky,kz,ksquared,weight)
#endif
  for(int ix=0;ix<local_n0_return;ix++)
     for(int iy=0;iy<nfft_dim;iy++)
        for(int iz=0;iz<=nfft_dim/2;iz++){
          kx = (ix+local_0_start_return>nfft_dim/2) ? 
              ix + local_0_start_return - nfft_dim : ix + local_0_start_return;
          ky = (iy > nfft_dim/2) ? iy-nfft_dim : iy;
          kz = (iz > nfft_dim/2) ? iz-nfft_dim : iz;
          ksquared = kx*kx + ky*ky + kz*kz;
          if (ksquared<=descriptor_kk_limit){
	    index1 = ix*N0_grid*(N0_grid/2+1) + iy*(N0_grid/2+1) + iz;
            weight = cabs(return_field[index1]);
            return_field[index1] /= weight;   
	  }; 	
	};

 };
 
printf("Reached here 12!\n");


if (nfft_dim <128){

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  char filename[100];
  sprintf(filename,"output_k_space_field.%d",rank);
  
  int xuse,yuse,zuse;
  FFTW_REAL sign;
  
  FILE *fp;

  fp = fopen(filename,"w");
  
  for (int ix=0; ix<local_n0_return; ix++)
    for (int iy=0; iy < nfft_dim; iy++)
      for (int iz=0; iz < nfft_dim; iz++){


        if (iz>nfft_dim/2){
          xuse = (nfft_dim-ix)%nfft_dim;
          yuse = (nfft_dim-iy)%nfft_dim;
          zuse = (nfft_dim-iz)%nfft_dim;
          sign = -1.0;
	}else{
          xuse = ix;
          yuse = iy;  
          zuse = iz;
          sign = 1.0;
        };

	int index = xuse*N0_grid*(N0_grid/2+1) + yuse*(N0_grid/2+1) + zuse;
        fprintf(fp,"%6d%6d%6d %14.8lf %14.8lf\n",ix+local_0_start_return,iy,iz,
		creal(return_field[index]),cimag(sign*return_field[index]));
		      };
  fclose(fp);



 };


 printf("Reached here 14!\n");

for (int j=0; j<4; j++){
   for (int i=0; i<4; i++) printf("(%lf  %lf) ",creal(return_field[j+ i*ncopy]),
				cimag(return_field[j + i*ncopy])); printf("\n");
 };



FFTW_FREE(output_coefficients);
FFTW_FREE(sph_bessel_coeff); 


 FFTW_DESTROY_PLAN(output_coeff_forward_plan);

 printf("Reached here! 3 \n");
 return(0);


 };


//==========================================================================================
//==========================================================================================





