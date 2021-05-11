
/////////////////////////////////////////////////
//  By default Panphasia is computed at single
//  precision.  To override this define PAN_DOUBLE


#ifndef USE_PRECISION_FLOAT
#define PAN_DOUBLE_PRECISION  8
#endif

#ifndef PAN_DOUBLE_PRECISION

#define PAN_REAL float
#define PAN_COMPLEX float complex

#else 

#define PAN_REAL double
#define PAN_COMPLEX  double complex

#endif




/////////////////////////////////////////////////////////////////////



void return_uniform_pseudo_rands_threefry4x64_(size_t l,size_t j1,size_t j2,size_t j3,
					       PAN_REAL *panphasia_randoms, size_t seed_value,
					       size_t allow_non_zero_seed_safety_catch);

void box_muller_(PAN_REAL *unif_real,PAN_REAL *gvar);

void solve_panphasia_cell_(PAN_REAL *input_vec_parent, PAN_REAL *input_vec_children, PAN_REAL *output_cell_vec, int control_flag);

void threefry4x64_test_(int verbose);
void inverse_threefry4x64_test_(int verbose);
void set_panphasia_key_(int verbose);
void check_panphasia_key_(int verbose);

void PANPHASIA_init_descriptor_checks();

void speed_test_();
void speed_test2_();
void check_randoms_();
void test_random_dist_(size_t shift);
void compute_all_properties_of_a_panphasia_cell_(size_t *level, size_t *j1, size_t *j2, size_t *j3,
					   PAN_REAL *gauss_rand_parent, PAN_REAL *legendre_rand);
void return_root_legendre_coefficients_(PAN_REAL *root);


int parse_and_validate_descriptor_(const char *, int *);
int demo_descriptor_();
long long int compute_check_digit_();
int PANPHASIA_init_descriptor_(char *descriptor, int *verbose);
int PANPHASIA_init_level_(size_t *oct_level, size_t *rel_orig_x, size_t *rel_orig_y,size_t *rel_orig_z,int *verbose);


int PANPHASIA_compute_coefficients_(size_t *xstart, size_t *ystart, size_t*zstart,
			  size_t *xextent, size_t *yextent, size_t *zextend,
			  size_t *copy_list, 
				    size_t *ncopy, void *output_values, int *flag_output_mode, int *verbose);

void test_moments_();
void test_propogation_of_moments_(int iterations);
void test_cell_moments(char *,size_t, size_t, size_t, size_t, size_t, double *);

void spherical_bessel_(int *, double *, double *);






void calc_absolute_coordinates(size_t xrel, size_t yrel, size_t zrel,size_t *xabs, size_t *yabs,size_t *zabs);

int cell_information(size_t cell_id, size_t *cumulative_cell_index, size_t *cuboid_x_dimen,
		      size_t *cuboid_y_dimen,size_t *cuboid_z_dimen, size_t *cell_lev,
                      size_t *cell_x, size_t *cell_y, size_t *cell_z, size_t number_children,
                      size_t *child_cell_indices);

int return_binary_tree_cell_lists(size_t level_max, size_t *list_cell_coordinates, 
				   size_t extent, size_t *return_tree_list_coordinates, 
				  size_t nreturn,
				  long long int *child_pointer, size_t *level_count, size_t *level_begin, size_t *index_perm);





void compute_sph_bessel_coeffs(int, int, int, int, double complex *);

int PANPHASIA_compute_kspace_field_(size_t, ptrdiff_t, ptrdiff_t, ptrdiff_t, FFTW_COMPLEX *);
