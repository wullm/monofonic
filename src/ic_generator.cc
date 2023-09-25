// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2020 by Oliver Hahn
// 
// monofonIC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// monofonIC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <general.hh>
#include <grid_fft.hh>
#include <operators.hh>
#include <convolution.hh>
#include <testing.hh>

#include <ic_generator.hh>
#include <particle_generator.hh>
#include <particle_plt.hh>

#include <unistd.h> // for unlink

#ifdef USE_FASTDF
#include <fastdf_cpp.h>
#endif

/**
 * @brief the possible species of fluids
 *  
 */
std::map<cosmo_species,std::string> cosmo_species_name = 
{
  {cosmo_species::dm,"Dark matter"},
  {cosmo_species::baryon,"Baryons"},
  {cosmo_species::neutrino,"Neutrinos"} // not implemented yet
};

/**
 * @brief the namespace encapsulating the main IC generation routines
 * 
 */
namespace ic_generator{

//! global RNG object
std::unique_ptr<RNG_plugin> the_random_number_generator;

//! global output object
std::unique_ptr<output_plugin> the_output_plugin;

//! global cosmology object (calculates all things cosmological)
std::unique_ptr<cosmology::calculator>  the_cosmo_calc;

/**
 * @brief Initialises all global objects
 * 
 * @param the_config reference to config_file object
 * @return int 0 if successful
 */
int initialise( config_file& the_config )
{
    the_random_number_generator = std::move(select_RNG_plugin(the_config));
    the_cosmo_calc              = std::make_unique<cosmology::calculator>(the_config);
    the_output_plugin           = std::move(select_output_plugin(the_config, the_cosmo_calc));
    
    return 0;
}

/**
 * @brief Reset all global objects
 * 
 */
void reset () {
    the_random_number_generator.reset();
    the_output_plugin.reset();
    the_cosmo_calc.reset();
}


/**
 * @brief Main driver routine for IC generation, everything interesting happens here
 * 
 * @param the_config reference to the config_file object
 * @return int 0 if successful
 */
int run( config_file& the_config )
{
    //--------------------------------------------------------------------------------------------------------
    // Read run parameters
    //--------------------------------------------------------------------------------------------------------

    //--------------------------------------------------------------------------------------------------------
    //! number of resolution elements per dimension
    const size_t ngrid = the_config.get_value<size_t>("setup", "GridRes");

    //--------------------------------------------------------------------------------------------------------
    //! box side length in h-1 Mpc
    const real_t boxlen = the_config.get_value<double>("setup", "BoxLength");

    //--------------------------------------------------------------------------------------------------------
    //! starting redshift
    const real_t zstart = the_config.get_value<double>("setup", "zstart");

    //--------------------------------------------------------------------------------------------------------
    //! order of the LPT approximation 
    const int LPTorder = the_config.get_value_safe<double>("setup","LPTorder",100);

    //--------------------------------------------------------------------------------------------------------
    //! initialice particles on a bcc or fcc lattice instead of a standard sc lattice (doubles and quadruples the number of particles) 
    std::string lattice_str = the_config.get_value_safe<std::string>("setup","ParticleLoad","sc");
    const particle::lattice lattice_type = 
          ((lattice_str=="bcc")? particle::lattice_bcc 
        : ((lattice_str=="fcc")? particle::lattice_fcc 
        : ((lattice_str=="rsc")? particle::lattice_rsc 
        : ((lattice_str=="glass")? particle::lattice_glass
        : particle::lattice_sc))));

    //--------------------------------------------------------------------------------------------------------
    //! apply fixing of the complex mode amplitude following Angulo & Pontzen (2016) [https://arxiv.org/abs/1603.05253]
    const bool bDoFixing    = the_config.get_value_safe<bool>("setup", "DoFixing", false);
    const bool bDoInversion = the_config.get_value_safe<bool>("setup", "DoInversion", false);
    

    //--------------------------------------------------------------------------------------------------------
    //! do baryon ICs?
    const bool bDoBaryons = the_config.get_value_safe<bool>("setup", "DoBaryons", false );
    //! enable also back-scaled decaying relative velocity mode? only first order!
    const bool bDoLinearBCcorr = the_config.get_value_safe<bool>("setup", "DoBaryonVrel", false);
    // compute mass fractions 
    std::map< cosmo_species, double > Omega;
    if( bDoBaryons ){
        double Om = the_cosmo_calc->cosmo_param_["Omega_m"];
        double Ob = the_cosmo_calc->cosmo_param_["Omega_b"];
        Omega[cosmo_species::dm] = Om-Ob;
        Omega[cosmo_species::baryon] = Ob;
    }else{
        double Om = the_cosmo_calc->cosmo_param_["Omega_m"];
        Omega[cosmo_species::dm] = Om;
        Omega[cosmo_species::baryon] = 0.0;
    }

    music::ilog << "Using Omega_m = " << the_cosmo_calc->cosmo_param_["Omega_m"] << std::endl;
    music::ilog << "Using Omega_c = " << the_cosmo_calc->cosmo_param_["Omega_c"] << std::endl;
    music::ilog << "Using Omega_b = " << the_cosmo_calc->cosmo_param_["Omega_b"] << std::endl;
    music::ilog << "Using Omega_dm_parts = " << Omega[cosmo_species::dm] << std::endl;
    music::ilog << "Using Omega_b_parts = " << Omega[cosmo_species::baryon] << std::endl;

    //! master switch to activate the following neutrino corrections by default
    const bool bWithNeutrinos = the_config.get_value_safe<bool>("setup", "WithNeutrinos", false );

    //! do not include massive neutrinos in the cdm fluid (use Omega_cdm not Omega_cdm + Omega_nu_massive)
    const bool bExcludeNeutrinos = the_config.get_value_safe<bool>("setup", "ExcludeNeutrinos", bWithNeutrinos );
    //! option to exclude massive neutrinos from delta_matter
    const bool bCDMBaryonMatterOnly = the_config.get_value_safe<bool>("setup", "CDMBaryonMatterOnly", bWithNeutrinos );
    //! option to do the second order neutrino correction
    const bool bDoNeutrinoPhi2Corr = the_config.get_value_safe<bool>("setup", "DoNeutrinoPhi2Corr", bWithNeutrinos );
    //! option to do the third order neutrino correction
    const bool bDoNeutrinoPhi3Corr = the_config.get_value_safe<bool>("setup", "DoNeutrinoPhi3Corr", bWithNeutrinos );

    //! correct for the difference between delta_matter and theta_matter on large scales
    const bool bDoDensityVelocityCorr = the_config.get_value_safe<bool>("setup", "DoDensityVelocityCorr", false );

    //! which transfer function plugin was used
    std::string tf = the_config.get_value<std::string>("cosmology", "transfer");
    if (bWithNeutrinos && tf != "zwindstroom") {
        music::wlog << " Neutrino corrections enabled. It is recommended to use the zwindstroom" << std::endl;
        music::wlog << " transfer function plugin for correct scale-dependent growth factors." << std::endl;
    }

    //! if necessary, subtract the massive neutrino density
    if (bExcludeNeutrinos) {
        double Onu = the_cosmo_calc->cosmo_param_["Omega_nu_massive"];
        Omega[cosmo_species::dm] -= Onu;
    }

    //! Do neutrino particle ICs?
    //! Note that this is distinct from the WithNeutrinos switch, which only affects CDM & Baryon ICs.
    const bool bDoNeutrinoParts = the_config.get_value_safe<bool>("setup", "DoNeutrinoParticles", false );
    const size_t NeutrinoCubeRootNum = the_config.get_value_safe<size_t>("setup", "NeutrinoCubeRootNum", 0 );
    const size_t NeutrinoGridRes = the_config.get_value_safe<size_t>("setup", "NeutrinoGridRes", 0 );
    const real_t NeutrinoStepSize = the_config.get_value_safe<real_t>("setup", "NeutrinoStepSize", 0.05 );
    const real_t NeutrinoInterpOrder = the_config.get_value_safe<int>("setup", "NeutrinoInterpOrder", 2 );
    const size_t TotalNeutrinoNum = NeutrinoCubeRootNum * NeutrinoCubeRootNum * NeutrinoCubeRootNum;
    const std::string white_noise_fname = "white_noise.hdf5";
    const std::string white_noise_dset = "white_noise";

    //--------------------------------------------------------------------------------------------------------
    //! do constrained ICs?
    const bool bAddConstrainedModes =  the_config.contains_key("random", "ConstraintFieldFile" );

    //--------------------------------------------------------------------------------------------------------
    //! add beyond box tidal field modes following Schmidt et al. (2018) [https://arxiv.org/abs/1803.03274]
    bool bAddExternalTides = the_config.contains_key("cosmology", "LSS_aniso_lx") 
                           & the_config.contains_key("cosmology", "LSS_aniso_ly") 
                           & the_config.contains_key("cosmology", "LSS_aniso_lz");

    if( bAddExternalTides && !(  the_config.contains_key("cosmology", "LSS_aniso_lx") 
                               | the_config.contains_key("cosmology", "LSS_aniso_ly") 
                               | the_config.contains_key("cosmology", "LSS_aniso_lz") ))
    {
        music::elog << "Not all dimensions of LSS_aniso_l{x,y,z} specified! Will ignore external tidal field!" << std::endl;
        bAddExternalTides = false;
    }

    if( bAddExternalTides && LPTorder == 1 ){
        music::elog << "External tidal field requires 2LPT! Will ignore external tidal field!" << std::endl;
        bAddExternalTides = false;
    }

    if( bAddExternalTides && LPTorder > 2 ){
        music::elog << "External tidal field requires 2LPT! Use >2LPT at your own risk (not proven to be correct)." << std::endl;
    }

    // Anisotropy parameters for beyond box tidal field 
    const std::array<real_t,3> lss_aniso_lambda = {
        real_t(the_config.get_value_safe<double>("cosmology", "LSS_aniso_lx", 0.0)),
        real_t(the_config.get_value_safe<double>("cosmology", "LSS_aniso_ly", 0.0)),
        real_t(the_config.get_value_safe<double>("cosmology", "LSS_aniso_lz", 0.0)),
    };  
    
    const real_t lss_aniso_sum_lambda = lss_aniso_lambda[0]+lss_aniso_lambda[1]+lss_aniso_lambda[2];

    //--------------------------------------------------------------------------------------------------------

    const real_t astart = 1.0/(1.0+zstart);
    const real_t volfac(std::pow(boxlen / ngrid / 2.0 / M_PI, 1.5));

    the_cosmo_calc->write_powerspectrum(astart, "input_powerspec.txt" );
    the_cosmo_calc->write_transfer("input_transfer.txt" );

    // the_cosmo_calc->compute_sigma_bc();
    // abort();

    //--------------------------------------------------------------------
    // Compute LPT time coefficients
    //--------------------------------------------------------------------
    const real_t Dplus0 = the_cosmo_calc->get_growth_factor(astart);
    const real_t vfac   = the_cosmo_calc->get_vfact_start();

    music::ilog << "Using D+ = " << Dplus0 << ", vfac = " << vfac << std::endl;

    const real_t g1  = -Dplus0;
    const real_t g2  = ((LPTorder>1)? -3.0/7.0*Dplus0*Dplus0 : 0.0);
    const real_t g3  = ((LPTorder>2)? 1.0/3.0*Dplus0*Dplus0*Dplus0 : 0.0);
    const real_t g3c = ((LPTorder>2)? 1.0/7.0*Dplus0*Dplus0*Dplus0 : 0.0);

    // vfac = d log D+ / dt 
    // d(D+^2)/dt = 2*D+ * d D+/dt = 2 * D+^2 * vfac
    // d(D+^3)/dt = 3*D+^2* d D+/dt = 3 * D+^3 * vfac
    const real_t vfac1 =  vfac;
    const real_t vfac2 =  2*vfac;
    const real_t vfac3 =  3*vfac;

    // anisotropic velocity growth factor for external tides
    // cf. eq. (5) of Stuecker et al. 2020 (https://arxiv.org/abs/2003.06427)
    const std::array<real_t,3> lss_aniso_alpha = {
        real_t(1.0) - Dplus0 * lss_aniso_lambda[0],
        real_t(1.0) - Dplus0 * lss_aniso_lambda[1],
        real_t(1.0) - Dplus0 * lss_aniso_lambda[2],
    };

    //--------------------------------------------------------------------
    // Create arrays
    //--------------------------------------------------------------------

    // white noise field 
    Grid_FFT<real_t> wnoise({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    //... Fill the wnoise grid with a Gaussian white noise field, we do this first since the RNG might need extra memory
    music::ilog << "-------------------------------------------------------------------------------" << std::endl;
    music::ilog << "Generating white noise field...." << std::endl;

    the_random_number_generator->Fill_Grid(wnoise);
    
    wnoise.FourierTransformForward();

    //... Next, declare LPT related arrays, allocated only as needed by order
    Grid_FFT<real_t> phi({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> phi2({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}, false); // do not allocate these unless needed
    Grid_FFT<real_t> phi3({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}, false); //   ..
    Grid_FFT<real_t> A3x({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}, false);  //   ..
    Grid_FFT<real_t> A3y({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}, false);  //   ..
    Grid_FFT<real_t> A3z({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}, false);  //   ..

    //... array [.] access to components of A3:
    std::array<Grid_FFT<real_t> *, 3> A3({&A3x, &A3y, &A3z});

    // temporary storage of additional data
    Grid_FFT<real_t> tmp({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    //--------------------------------------------------------------------
    // Use externally specified large scale modes from constraints in case
    // TODO: move to separate routine
    //--------------------------------------------------------------------
    if( bAddConstrainedModes ){
        Grid_FFT<real_t,false> cwnoise({8,8,8}, {boxlen,boxlen,boxlen});
        cwnoise.Read_from_HDF5( the_config.get_value<std::string>("random", "ConstraintFieldFile"), 
                the_config.get_value<std::string>("random", "ConstraintFieldName") );
        cwnoise.FourierTransformForward();

        size_t ngrid_c = cwnoise.size(0), ngrid_c_2 = ngrid_c/2;

        // TODO: copy over modes
        double rs1{0.0},rs2{0.0},is1{0.0},is2{0.0};
        double nrs1{0.0},nrs2{0.0},nis1{0.0},nis2{0.0};
        size_t count{0};

        #pragma omp parallel for reduction(+:rs1,rs2,is1,is2,nrs1,nrs2,nis1,nis2,count)
        for( size_t i=0; i<ngrid_c; ++i ){
            size_t il = size_t(-1);
            if( i<ngrid_c_2 && i<ngrid/2 ) il = i;
            if( i>=ngrid_c_2 && i+ngrid-ngrid_c>=ngrid/2) il = ngrid-ngrid_c+i;
            if( il == size_t(-1) ) continue;
            if( il<size_t(wnoise.local_1_start_) || il>=size_t(wnoise.local_1_start_+wnoise.local_1_size_)) continue;
            il -= wnoise.local_1_start_;
            for( size_t j=0; j<ngrid_c; ++j ){
                size_t jl = size_t(-1);
                if( j<ngrid_c_2 && j<ngrid/2 ) jl = j;
                if( j>=ngrid_c_2 && j+ngrid-ngrid_c>=ngrid/2 ) jl = ngrid-ngrid_c+j;
                if( jl == size_t(-1) ) continue;
                for( size_t k=0; k<ngrid_c/2+1; ++k ){
                    if( k>ngrid/2 ) continue;
                    size_t kl = k;
                    
                    ++count;

                    nrs1 += std::real(cwnoise.kelem(i,j,k));
                    nrs2 += std::real(cwnoise.kelem(i,j,k))*std::real(cwnoise.kelem(i,j,k));
                    nis1 += std::imag(cwnoise.kelem(i,j,k));
                    nis2 += std::imag(cwnoise.kelem(i,j,k))*std::imag(cwnoise.kelem(i,j,k));

                    rs1 += std::real(wnoise.kelem(il,jl,kl));
                    rs2 += std::real(wnoise.kelem(il,jl,kl))*std::real(wnoise.kelem(il,jl,kl));
                    is1 += std::imag(wnoise.kelem(il,jl,kl));
                    is2 += std::imag(wnoise.kelem(il,jl,kl))*std::imag(wnoise.kelem(il,jl,kl));
                    
                #if defined(USE_MPI)
                    wnoise.kelem(il,jl,kl) = cwnoise.kelem(j,i,k);
                #else
                    wnoise.kelem(il,jl,kl) = cwnoise.kelem(i,j,k);
                #endif
                }
            }
        }

        // music::ilog << "  ... old field: re <w>=" << rs1/count << " <w^2>-<w>^2=" << rs2/count-rs1*rs1/count/count << std::endl;
        // music::ilog << "  ... old field: im <w>=" << is1/count << " <w^2>-<w>^2=" << is2/count-is1*is1/count/count << std::endl;
        // music::ilog << "  ... new field: re <w>=" << nrs1/count << " <w^2>-<w>^2=" << nrs2/count-nrs1*nrs1/count/count << std::endl;
        // music::ilog << "  ... new field: im <w>=" << nis1/count << " <w^2>-<w>^2=" << nis2/count-nis1*nis1/count/count << std::endl;
        music::ilog << "White noise field large-scale modes overwritten with external field." << std::endl;
    }

    //--------------------------------------------------------------------
    // Apply Normalisation factor and Angulo&Pontzen fixing or not
    //--------------------------------------------------------------------

    wnoise.apply_function_k( [&](auto wn){
        if (bDoFixing){
            wn = (std::fabs(wn) != 0.0) ? wn / std::fabs(wn) : wn;
        }
        return ((bDoInversion)? real_t{-1.0} : real_t{1.0}) * wn / volfac;
    });

    //--------------------------------------------------------------------
    // Potentially, create a down-sampled copy of the random phases
    //--------------------------------------------------------------------
    if (bDoNeutrinoParts) {
        const size_t nsmall = NeutrinoGridRes;
        const size_t nsmall_2 = nsmall / 2;

        /* Can we export the white noise field as is or do we need to downsample? */
        if (nsmall == ngrid) {
            wnoise.FourierTransformBackward();
#if defined(USE_MPI)
            if (CONFIG::MPI_task_rank == 0)
                unlink(white_noise_fname.c_str());
            MPI_Barrier(MPI_COMM_WORLD);
#else
            unlink(white_noise_fname.c_str());
#endif
            wnoise.Write_to_HDF5(white_noise_fname, white_noise_dset);

            wnoise.FourierTransformForward();
        } else if (nsmall > 0 && nsmall < ngrid) {

            Grid_FFT<real_t,false> noise_small({nsmall,nsmall,nsmall}, {boxlen,boxlen,boxlen});
            noise_small.FourierTransformForward();
            noise_small.apply_function_k( [&](auto wn){
                return 0.0;
            });

            #pragma omp parallel for
            for( size_t i=0; i<nsmall; ++i ){
                size_t il = size_t(-1);
                if( i<nsmall_2 && i<ngrid/2 ) il = i;
                if( i>=nsmall_2 && i+ngrid-nsmall>=ngrid/2) il = ngrid-nsmall+i;
                if( il == size_t(-1) ) continue;
                if( il<size_t(wnoise.local_1_start_) || il>=size_t(wnoise.local_1_start_+wnoise.local_1_size_)) continue;
                il -= wnoise.local_1_start_;
                for( size_t j=0; j<nsmall; ++j ){
                    size_t jl = size_t(-1);
                    if( j<nsmall_2 && j<ngrid/2 ) jl = j;
                    if( j>=nsmall_2 && j+ngrid-nsmall>=ngrid/2 ) jl = ngrid-nsmall+j;
                    if( jl == size_t(-1) ) continue;
                    for( size_t k=0; k<nsmall/2+1; ++k ){
                        if( k>ngrid/2 ) continue;
                        size_t kl = k;

                    #if defined(USE_MPI)
                        noise_small.kelem(j,i,k) = wnoise.kelem(il,jl,kl);
                    #else
                        noise_small.kelem(i,j,k) = wnoise.kelem(il,jl,kl);
                    #endif
                    }
                }
            }

            /* Make sure that the normalization is correct */
            const real_t volfac_small(std::pow(boxlen / nsmall / 2.0 / M_PI, 1.5));
            noise_small.apply_function_k( [&](auto wn){
                return wn * volfac / volfac_small;
            });

            noise_small.FourierTransformBackward();

#if defined(USE_MPI)
            Grid_FFT<real_t,false> noise_small_aggr({nsmall,nsmall,nsmall}, {boxlen,boxlen,boxlen});
            MPI_Reduce(noise_small.data_, noise_small_aggr.data_, nsmall*nsmall*nsmall,
                       MPI::get_datatype<real_t>(), MPI_SUM, 0, MPI_COMM_WORLD);
            if (CONFIG::MPI_task_rank == 0) {
                unlink(white_noise_fname.c_str());
                noise_small_aggr.Write_to_HDF5(white_noise_fname, white_noise_dset);
            }
#else
            unlink(white_noise_fname.c_str());
            noise_small.Write_to_HDF5(white_noise_fname, white_noise_dset);
#endif
        } else if (nsmall <= 0) {
            throw std::runtime_error("The neutrino grid size should be a positive integer.");
        } else {
            throw std::runtime_error("Cannot upsample the white noise field.");
        }
    }


    //--------------------------------------------------------------------
    // Compute the LPT terms....
    //--------------------------------------------------------------------

    //--------------------------------------------------------------------
    // Create convolution class instance for non-linear terms
    //--------------------------------------------------------------------
#if defined(USE_CONVOLVER_ORSZAG)
    OrszagConvolver<real_t> Conv({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
#elif defined(USE_CONVOLVER_NAIVE)
    NaiveConvolver<real_t> Conv({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
#endif
    //--------------------------------------------------------------------

    //--------------------------------------------------------------------
    // Create PLT gradient operator
    //--------------------------------------------------------------------
#if defined(ENABLE_PLT)
    particle::lattice_gradient lg( the_config );
#else
    op::fourier_gradient lg( the_config );
#endif

    //--------------------------------------------------------------------
    std::vector<cosmo_species> species_list;
    species_list.push_back(cosmo_species::dm);
    if (bDoBaryons)
        species_list.push_back(cosmo_species::baryon);
    if (bDoNeutrinoParts)
        species_list.push_back(cosmo_species::neutrino);

    //======================================================================
    //... compute 1LPT displacement potential ....
    //======================================================================
    // phi = - delta / k^2

    music::ilog << "-------------------------------------------------------------------------------" << std::endl;
    music::ilog << "Generating LPT fields...." << std::endl;

    double wtime = get_wtime();
    music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(1) term" << std::flush;

    phi.FourierTransformForward(false);
    phi.assign_function_of_grids_kdep([&](auto k, auto wn) {
        real_t kmod = k.norm();
        ccomplex_t delta = wn * the_cosmo_calc->get_amplitude(kmod, delta_matter);
        return -delta / (kmod * kmod);
    }, wnoise);

    phi.zero_DC_mode();

    music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime() - wtime << "s" << std::endl;

    //======================================================================
    //... compute 2LPT displacement potential ....
    //======================================================================
    if (LPTorder > 1)
    {
        phi2.allocate();
        phi2.FourierTransformForward(false);
        
        wtime = get_wtime();
        music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(2) term" << std::flush;
        Conv.convolve_SumOfHessians(phi, {0, 0}, phi, {1, 1}, {2, 2}, op::assign_to(phi2));
        Conv.convolve_Hessians(phi, {1, 1}, phi, {2, 2}, op::add_to(phi2));
        Conv.convolve_Hessians(phi, {0, 1}, phi, {0, 1}, op::subtract_from(phi2));
        Conv.convolve_Hessians(phi, {0, 2}, phi, {0, 2}, op::subtract_from(phi2));
        Conv.convolve_Hessians(phi, {1, 2}, phi, {1, 2}, op::subtract_from(phi2));

        if (bAddExternalTides)
        {
            // anisotropic contribution to Phi^{(2)} for external tides, note that phi2 = nabla^2 phi^(2) at this point.
            // cf. eq. (19) of Stuecker et al. 2020 (https://arxiv.org/abs/2003.06427)
            phi2.assign_function_of_grids_kdep([&](vec3_t<real_t> kvec, ccomplex_t pphi, ccomplex_t pphi2) {
                real_t k2 = kvec.norm_squared();
                real_t fac_aniso = (kvec[0] * kvec[0] * lss_aniso_lambda[0] + kvec[1] * kvec[1] * lss_aniso_lambda[1] + kvec[2] * kvec[2] * lss_aniso_lambda[2]);
                return pphi2 - (lss_aniso_sum_lambda * k2 + real_t(4.0/3.0) * fac_aniso ) * pphi;
            }, phi, phi2);
        }

        phi2.apply_InverseLaplacian();
        music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime() - wtime << "s" << std::endl;

        if (bAddExternalTides)
        {
            music::wlog << "Added external tide contribution to phi(2)... Make sure your N-body code supports this!" << std::endl;
            music::wlog << " lss_aniso = (" << lss_aniso_lambda[0] << ", " << lss_aniso_lambda[1] << ", " << lss_aniso_lambda[2] << ")" << std::endl;
        }
    }

    //! analytical neutrino correction factor for phi2
    if (LPTorder > 1 && bDoNeutrinoPhi2Corr) {
        const real_t O_m = the_cosmo_calc->cosmo_param_["Omega_m"];
        const real_t f_nu = the_cosmo_calc->cosmo_param_["Omega_nu_massive"] / O_m;
        const real_t C_2 = 14. * (1. - f_nu) / (19. - 18. * f_nu - sqrt(25. - 24. * f_nu));

        music::ilog << "Rescaling phi(2) by " << C_2 << std::endl;

        phi2 *= C_2;
    }

    //======================================================================
    //... compute 3LPT displacement potential
    //======================================================================
    if (LPTorder > 2)
    {
        phi3.allocate();
        phi3.FourierTransformForward(false);

        
        //... phi3 = phi3a - 10/7 phi3b
        //... 3a term ...
        wtime = get_wtime();
        music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(3a) term" << std::flush;
        Conv.convolve_Hessians(phi, {0, 0}, phi, {1, 1}, phi, {2, 2}, op::assign_to(phi3));
        Conv.convolve_Hessians(phi, {0, 1}, phi, {0, 2}, phi, {1, 2}, op::multiply_add_to(phi3,2.0));
        Conv.convolve_Hessians(phi, {1, 2}, phi, {1, 2}, phi, {0, 0}, op::subtract_from(phi3));
        Conv.convolve_Hessians(phi, {0, 2}, phi, {0, 2}, phi, {1, 1}, op::subtract_from(phi3));
        Conv.convolve_Hessians(phi, {0, 1}, phi, {0, 1}, phi, {2, 2}, op::subtract_from(phi3));
        // phi3a.apply_InverseLaplacian();
        music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime() - wtime << "s" << std::endl;

        //! analytical neutrino correction factor for phi3a
        if (bDoNeutrinoPhi3Corr) {
            // We multiply phi3a by C_1 now and divide (phi3a + phi3b) by C_1 later
            const real_t O_m = the_cosmo_calc->cosmo_param_["Omega_m"];
            const real_t f_nu = the_cosmo_calc->cosmo_param_["Omega_nu_massive"] / O_m;
            const real_t C_1 = 20. * (1 - f_nu) / (25. - 24. * f_nu - sqrt(25. - 24. * f_nu));

            music::ilog << "Rescaling phi(3a) by " << C_1 << std::endl;

            phi3 *= C_1;
        }

        //... 3b term ...
        wtime = get_wtime();
        music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(3b) term" << std::flush;
        // phi3b.FourierTransformForward(false);
        Conv.convolve_SumOfHessians(phi, {0, 0}, phi2, {1, 1}, {2, 2}, op::multiply_add_to(phi3,-5.0/7.0));
        Conv.convolve_SumOfHessians(phi, {1, 1}, phi2, {2, 2}, {0, 0}, op::multiply_add_to(phi3,-5.0/7.0));
        Conv.convolve_SumOfHessians(phi, {2, 2}, phi2, {0, 0}, {1, 1}, op::multiply_add_to(phi3,-5.0/7.0));
        Conv.convolve_Hessians(phi, {0, 1}, phi2, {0, 1}, op::multiply_add_to(phi3,+10.0/7.0));
        Conv.convolve_Hessians(phi, {0, 2}, phi2, {0, 2}, op::multiply_add_to(phi3,+10.0/7.0));
        Conv.convolve_Hessians(phi, {1, 2}, phi2, {1, 2}, op::multiply_add_to(phi3,+10.0/7.0));
        phi3.apply_InverseLaplacian();
        //phi3b *= 0.5; // factor 1/2 from definition of phi(3b)!
        music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime() - wtime << "s" << std::endl;

        //... transversal term ...
        wtime = get_wtime();
        music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing A(3) term" << std::flush;
        for (int idim = 0; idim < 3; ++idim)
        {
            // cyclic rotations of indices
            int idimp = (idim + 1) % 3, idimpp = (idim + 2) % 3;
            A3[idim]->allocate();
            A3[idim]->FourierTransformForward(false);
            Conv.convolve_Hessians(phi2, {idim, idimp}, phi, {idim, idimpp}, op::assign_to(*A3[idim]));
            Conv.convolve_Hessians(phi2, {idim, idimpp}, phi, {idim, idimp}, op::subtract_from(*A3[idim]));
            Conv.convolve_DifferenceOfHessians(phi, {idimp, idimpp}, phi2, {idimp, idimp}, {idimpp, idimpp}, op::add_to(*A3[idim]));
            Conv.convolve_DifferenceOfHessians(phi2, {idimp, idimpp}, phi, {idimp, idimp}, {idimpp, idimpp}, op::subtract_from(*A3[idim]));
            A3[idim]->apply_InverseLaplacian();
        }
        music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime() - wtime << "s" << std::endl;
    }

    //! analytical neutrino correction factor for phi3a and phi3b
    if (LPTorder > 2 && bDoNeutrinoPhi3Corr) {
        const real_t O_m = the_cosmo_calc->cosmo_param_["Omega_m"];
        const real_t f_nu = the_cosmo_calc->cosmo_param_["Omega_nu_massive"] / O_m;
        const real_t C_1 = 20. * (1 - f_nu) / (25. - 24. * f_nu - sqrt(25. - 24. * f_nu));
        const real_t C_3 = 12. * (1. - f_nu) / (17. - 16. * f_nu - sqrt(25. - 24. * f_nu));
        const real_t Phi3RescaleFact = C_3 / C_1;

        music::ilog << "Rescaling phi(3a) and phi(3b) by " << Phi3RescaleFact << std::endl;

        phi3 *= Phi3RescaleFact;
    }

    ///... scale all potentials with respective growth factors
    phi *= g1;

    if (LPTorder > 1)
    {
        phi2 *= g2;
    }
    
    if (LPTorder > 2)
    {
        phi3 *= g3;
        (*A3[0]) *= g3c;
        (*A3[1]) *= g3c;
        (*A3[2]) *= g3c;
    }

    music::ilog << "-------------------------------------------------------------------------------" << std::endl;

    ///////////////////////////////////////////////////////////////////////
    // we store the densities here if we compute them
    //======================================================================

    // Testing
    const std::string testing = the_config.get_value_safe<std::string>("testing", "test", "none");

    if (testing != "none")
    {
        music::wlog << "you are running in testing mode. No ICs, only diagnostic output will be written out!" << std::endl;
        if (testing == "white_noise"){
            testing::output_white_noise(the_config, ngrid, boxlen, wnoise);
        }
        else if (testing == "potentials_and_densities"){
            testing::output_potentials_and_densities(the_config, ngrid, boxlen, phi, phi2, phi3, A3);
        }
        else if (testing == "velocity_displacement_symmetries"){
            testing::output_velocity_displacement_symmetries(the_config, ngrid, boxlen, vfac, Dplus0, phi, phi2, phi3, A3);
        }
        else if (testing == "convergence"){
            testing::output_convergence(the_config, the_cosmo_calc.get(), ngrid, boxlen, vfac, Dplus0, phi, phi2, phi3, A3);
        }
        else{
            music::flog << "unknown test '" << testing << "'" << std::endl;
            std::abort();
        }
    }

    // // write out internally computed growth factor
    // if( true && MPI::get_rank()==0 )
    // {
    //     std::ofstream ofs("growthfac.txt");
    //     double a=1e-3;
    //     double ainc = 1.01;
    //     while( a<1.1 ){
    //         ofs << std::setw(16) << a << " " << std::setw(16) << the_cosmo_calc->get_growth_factor( a ) << std::endl;
    //         a *= ainc;
    //     }
    //     ofs.close();
    // }

    //==============================================================//
    // main output loop, loop over all species that are enabled
    //==============================================================//
    for( const auto& this_species : species_list )
    {
        music::ilog << std::endl
                    << ">>> Computing ICs for species \'" << cosmo_species_name[this_species] << "\' <<<\n" << std::endl;

        // total matter density at z=0, including massive neutrinos
        const real_t O_m = the_cosmo_calc->cosmo_param_["Omega_m"];
        const real_t O_c = the_cosmo_calc->cosmo_param_["Omega_c"];
        const real_t O_dcdm = the_cosmo_calc->cosmo_param_["Omega_dcdm_start"];
        const real_t O_dm = O_c + O_dcdm;

        // baryon, cdm, and massive neutrino fractions of the total matter density (z=0)
        const real_t f_b = the_cosmo_calc->cosmo_param_["Omega_b"] / O_m;
        const real_t f_c = O_dm / O_m;

        // C factor for baryons and cdm
        real_t C_species;
        if (bDoBaryons && bExcludeNeutrinos){
            C_species = (this_species == cosmo_species::baryon)? (1.0-f_b / (f_b + f_c)) : -f_b / (f_b + f_c);
        }else if (bDoBaryons && !bExcludeNeutrinos){
            C_species = (this_species == cosmo_species::baryon)? (1.0-f_b) : -f_b;
        }else{
            C_species = 0.;
        }

        // main loop block
        {
            std::unique_ptr<particle::lattice_generator<Grid_FFT<real_t>>> particle_lattice_generator_ptr;

            // if output plugin wants particles, then we need to store them, along with their IDs
            if( the_output_plugin->write_species_as( this_species ) == output_type::particles
                && this_species != cosmo_species::neutrino)
            {
                // somewhat arbitrarily, start baryon particle IDs from 2**31 if we have 32bit and from 2**56 if we have 64 bits
                size_t IDoffset = (this_species == cosmo_species::baryon)? ((the_output_plugin->has_64bit_ids())? 1 : 1): 0 ;

                // allocate particle structure and generate particle IDs
                particle_lattice_generator_ptr = 
                std::make_unique<particle::lattice_generator<Grid_FFT<real_t>>>( lattice_type, the_output_plugin->has_64bit_reals(), the_output_plugin->has_64bit_ids(), 
                    bDoBaryons, IDoffset, tmp, the_config );
            }

            // set the perturbed particle masses if we have baryons
            if( bDoBaryons && (the_output_plugin->write_species_as( this_species ) == output_type::particles
                || the_output_plugin->write_species_as( this_species ) == output_type::field_lagrangian)
                && this_species != cosmo_species::neutrino )
            {
                bool shifted_lattice = (this_species == cosmo_species::baryon &&
                                        the_output_plugin->write_species_as(this_species) == output_type::particles) ? true : false;

                const real_t munit = the_output_plugin->mass_unit();

                //======================================================================
                // initialise rho
                //======================================================================
                Grid_FFT<real_t> rho({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

                wnoise.FourierTransformForward();
                rho.FourierTransformForward(false);
                rho.assign_function_of_grids_kdep( [&]( auto k, auto wn ){
                    return wn * the_cosmo_calc->get_amplitude_delta_bc(k.norm(),bDoLinearBCcorr);
                }, wnoise );
                rho.zero_DC_mode();
                rho.FourierTransformBackward();

                rho.apply_function_r( [&]( auto prho ){
                    return (1.0 + C_species * prho) * Omega[this_species] * munit;
                });
                
                if( the_output_plugin->write_species_as( this_species ) == output_type::particles ){
                    particle_lattice_generator_ptr->set_masses( lattice_type, shifted_lattice, 1.0, the_output_plugin->has_64bit_reals(), rho, the_config );
                }else if( the_output_plugin->write_species_as( this_species ) == output_type::field_lagrangian ){
                    the_output_plugin->write_grid_data( rho, this_species, fluid_component::mass );
                }
            }

            //if( the_output_plugin->write_species_as( cosmo_species::dm ) == output_type::field_eulerian ){
            if( the_output_plugin->write_species_as(this_species) == output_type::field_eulerian )
            {
                //======================================================================
                // use QPT to get density and velocity fields
                //======================================================================
                Grid_FFT<ccomplex_t> psi({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
                Grid_FFT<real_t> rho({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

                //======================================================================
                // initialise rho
                //======================================================================
                wnoise.FourierTransformForward();
                rho.FourierTransformForward(false);
                rho.assign_function_of_grids_kdep( [&]( auto k, auto wn ){
                    return wn * the_cosmo_calc->get_amplitude_delta_bc(k.norm(), false);
                }, wnoise );
                rho.zero_DC_mode();
                rho.FourierTransformBackward();
                
                rho.apply_function_r( [&]( auto prho ){
                    return std::sqrt( 1.0 + C_species * prho );
                });

                //======================================================================
                // initialise psi = exp(i Phi(1)/hbar)
                //======================================================================
                phi.FourierTransformBackward();

                real_t maxdphi = -1.0;

                #pragma omp parallel for reduction(max:maxdphi)
                for( size_t i=0; i<phi.size(0)-1; ++i ){
                    size_t ir = (i+1)%phi.size(0);
                    for( size_t j=0; j<phi.size(1); ++j ){
                        size_t jr = (j+1)%phi.size(1);    
                        for( size_t k=0; k<phi.size(2); ++k ){
                            size_t kr = (k+1)%phi.size(2);
                            auto phic = phi.relem(i,j,k);

                            auto dphixr = std::fabs(phi.relem(ir,j,k) - phic);
                            auto dphiyr = std::fabs(phi.relem(i,jr,k) - phic);
                            auto dphizr = std::fabs(phi.relem(i,j,kr) - phic);
                            
                            maxdphi = std::max(maxdphi,std::max(dphixr,std::max(dphiyr,dphizr)));
                        }
                    }
                }
                #if defined(USE_MPI)
                    real_t local_maxdphi = maxdphi;
                    MPI_Allreduce( &local_maxdphi, &maxdphi, 1, MPI::get_datatype<real_t>(), MPI_MAX, MPI_COMM_WORLD );
                #endif
                const real_t hbar_safefac = 1.01;
                const real_t hbar = maxdphi / M_PI / Dplus0 * hbar_safefac;
                music::ilog << "Semiclassical PT : hbar = " << hbar << " (limited by initial potential, safety=" << hbar_safefac << ")." << std::endl;
                
                if( LPTorder == 1 ){
                    psi.assign_function_of_grids_r([hbar,Dplus0]( real_t pphi, real_t prho ){
                        return prho * std::exp(ccomplex_t(0.0,1.0/hbar) * (pphi / Dplus0)); // divide by Dplus since phi already contains it
                    }, phi, rho );
                }else if( LPTorder >= 2 ){
                    phi2.FourierTransformBackward();
                    // we don't have a 1/2 in the Veff term because pre-factor is already 3/7
                    psi.assign_function_of_grids_r([hbar,Dplus0]( real_t pphi, real_t pphi2, real_t prho ){
                        return prho * std::exp(ccomplex_t(0.0,1.0/hbar) * (pphi + pphi2) / Dplus0);
                    }, phi, phi2, rho );
                }

                //======================================================================
                // evolve wave-function (one drift step) psi = psi *exp(-i hbar *k^2 dt / 2)
                //======================================================================
                psi.FourierTransformForward();
                psi.apply_function_k_dep([hbar,Dplus0]( auto epsi, auto k ){
                    auto k2 = k.norm_squared();
                    return epsi * std::exp( - ccomplex_t(0.0,0.5)*hbar* k2 * Dplus0);
                });
                psi.FourierTransformBackward();

                if( LPTorder >= 2 ){
                    psi.assign_function_of_grids_r([&](auto ppsi, auto pphi2) {
                        return ppsi * std::exp(ccomplex_t(0.0,1.0/hbar) * (pphi2) / Dplus0);
                    }, psi, phi2);
                }

                //======================================================================
                // compute rho
                //======================================================================
                rho.assign_function_of_grids_r([&]( auto p ){
                    auto pp = std::real(p)*std::real(p) + std::imag(p)*std::imag(p) - 1.0;
                    return pp;
                }, psi);

                the_output_plugin->write_grid_data( rho, this_species, fluid_component::density );
                rho.Write_PowerSpectrum("input_powerspec_sampled_evolved_semiclassical.txt");
                rho.FourierTransformBackward();
                
                //======================================================================
                // compute  v
                //======================================================================
                Grid_FFT<ccomplex_t> grad_psi({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
                const real_t vunit = Dplus0 * vfac / boxlen * the_output_plugin->velocity_unit();
                for( int idim=0; idim<3; ++idim )
                {
                    grad_psi.FourierTransformBackward(false);
                    grad_psi.copy_from(psi);
                    grad_psi.FourierTransformForward();
                    grad_psi.apply_function_k_dep([&](auto x, auto k) {
                        return x * ccomplex_t(0.0,k[idim]);
                    });
                    grad_psi.FourierTransformBackward();
                    
                    tmp.FourierTransformBackward(false);
                    tmp.assign_function_of_grids_r([&](auto ppsi, auto pgrad_psi, auto prho) {
                            return vunit * std::real((std::conj(ppsi) * pgrad_psi - ppsi * std::conj(pgrad_psi)) / ccomplex_t(0.0, 2.0 / hbar)/real_t(1.0+prho));
                        }, psi, grad_psi, rho);

                    fluid_component fc = (idim==0)? fluid_component::vx : ((idim==1)? fluid_component::vy : fluid_component::vz );
                    the_output_plugin->write_grid_data( tmp, this_species, fc );
                }
            }

            if( (the_output_plugin->write_species_as( this_species ) == output_type::particles
             || the_output_plugin->write_species_as( this_species ) == output_type::field_lagrangian)
             && this_species != cosmo_species::neutrino )
            {
                //===================================================================================
                // we store displacements and velocities here if we compute them
                //===================================================================================
                

                bool shifted_lattice = (this_species == cosmo_species::baryon &&
                                        the_output_plugin->write_species_as(this_species) == output_type::particles) ? true : false;

                
                grid_interpolate<1,Grid_FFT<real_t>> interp( tmp );

                phi.FourierTransformForward();
                if( LPTorder > 1 ){
                    phi2.FourierTransformForward();
                }
                if( LPTorder > 2 ){
                    phi3.FourierTransformForward();
                    A3[0]->FourierTransformForward();
                    A3[1]->FourierTransformForward();
                    A3[2]->FourierTransformForward();
                }
                wnoise.FourierTransformForward();
            
                // write out positions
                for( int idim=0; idim<3; ++idim ){
                    // cyclic rotations of indices
                    const int idimp = (idim+1)%3, idimpp = (idim+2)%3;
                    const real_t lunit = the_output_plugin->position_unit();
                    
                    tmp.FourierTransformForward(false);

                    // combine the various LPT potentials into one and take gradient
                    #pragma omp parallel for
                    for (size_t i = 0; i < phi.size(0); ++i) {
                        for (size_t j = 0; j < phi.size(1); ++j) {
                            for (size_t k = 0; k < phi.size(2); ++k) {
                                size_t idx = phi.get_idx(i,j,k);
                                auto phitot = phi.kelem(idx);

                                if( LPTorder > 1 ){
                                    phitot += phi2.kelem(idx);
                                }

                                if( LPTorder > 2 ){
                                    phitot += phi3.kelem(idx);
                                }

                                tmp.kelem(idx) = lg.gradient(idim,tmp.get_k3(i,j,k)) * phitot;

                                if( LPTorder > 2 ){
                                    tmp.kelem(idx) += lg.gradient(idimp,tmp.get_k3(i,j,k)) * A3[idimpp]->kelem(idx) - lg.gradient(idimpp,tmp.get_k3(i,j,k)) * A3[idimp]->kelem(idx);
                                }

                                if( the_output_plugin->write_species_as( this_species ) == output_type::particles && lattice_type == particle::lattice_glass){
                                    tmp.kelem(idx) *= interp.compensation_kernel( tmp.get_k<real_t>(i,j,k) ) ;
                                }

                                // divide by Lbox, because displacement is in box units for output plugin
                                tmp.kelem(idx) *=  lunit / boxlen;
                            }
                        }
                    }
                    tmp.zero_DC_mode();
                    tmp.FourierTransformBackward();

                    // if we write particle data, store particle data in particle structure
                    if( the_output_plugin->write_species_as( this_species ) == output_type::particles )
                    {
                        particle_lattice_generator_ptr->set_positions( lattice_type, shifted_lattice, idim, lunit, the_output_plugin->has_64bit_reals(), tmp, the_config );
                    } 
                    // otherwise write out the grid data directly to the output plugin
                    // else if( the_output_plugin->write_species_as( cosmo_species::dm ) == output_type::field_lagrangian )
                    else if( the_output_plugin->write_species_as( this_species ) == output_type::field_lagrangian )
                    {
                        fluid_component fc = (idim==0)? fluid_component::dx : ((idim==1)? fluid_component::dy : fluid_component::dz );
                        the_output_plugin->write_grid_data( tmp, this_species, fc );
                    }
                }

                // write out velocities
                for( int idim=0; idim<3; ++idim ){
                    // cyclic rotations of indices
                    int idimp = (idim+1)%3, idimpp = (idim+2)%3;
                    const real_t vunit = the_output_plugin->velocity_unit();
                    
                    tmp.FourierTransformForward(false);

                    #pragma omp parallel for
                    for (size_t i = 0; i < phi.size(0); ++i) {
                        for (size_t j = 0; j < phi.size(1); ++j) {
                            for (size_t k = 0; k < phi.size(2); ++k) {
                                size_t idx = phi.get_idx(i,j,k);
                                
                                auto phitot_v = vfac1 * phi.kelem(idx);
                                
                                if( LPTorder > 1 ){
                                    phitot_v += vfac2 * phi2.kelem(idx);
                                }

                                if( LPTorder > 2 ){
                                    phitot_v += vfac3 * phi3.kelem(idx);
                                }
                                
                                tmp.kelem(idx) = lg.gradient(idim,tmp.get_k3(i,j,k)) * phitot_v;
                                
                                if( LPTorder > 2 ){
                                    tmp.kelem(idx) += vfac3 * (lg.gradient(idimp,tmp.get_k3(i,j,k)) * A3[idimpp]->kelem(idx) - lg.gradient(idimpp,tmp.get_k3(i,j,k)) * A3[idimp]->kelem(idx));
                                }

                                // if multi-species, then add vbc component backwards
                                if( bDoBaryons & bDoLinearBCcorr ){
                                    real_t knorm = wnoise.get_k<real_t>(i,j,k).norm();
                                    tmp.kelem(idx) -= vfac1 * C_species * the_cosmo_calc->get_amplitude_theta_bc(knorm, bDoLinearBCcorr) * wnoise.kelem(i,j,k) * lg.gradient(idim,tmp.get_k3(i,j,k)) / (knorm*knorm);
                                }

                                // option to account for the difference between delta_m and theta_m / (afH) on large scales
                                if (bDoDensityVelocityCorr) {
                                    real_t knorm = wnoise.get_k<real_t>(i,j,k).norm();
                                    tmp.kelem(idx) += vfac1 * the_cosmo_calc->get_amplitude_theta_delta_m(knorm, bCDMBaryonMatterOnly) * wnoise.kelem(i,j,k) * lg.gradient(idim,tmp.get_k3(i,j,k)) / (knorm*knorm);
                                }

                                // correct with interpolation kernel if we used interpolation to read out the positions (for glasses)
                                if( the_output_plugin->write_species_as( this_species ) == output_type::particles && lattice_type == particle::lattice_glass){
                                    tmp.kelem(idx) *= interp.compensation_kernel( tmp.get_k<real_t>(i,j,k) );
                                }

                                // correct velocity with PLT mode growth rate
                                tmp.kelem(idx) *= lg.vfac_corr(tmp.get_k3(i,j,k));

                                if( bAddExternalTides ){
                                    // modify velocities with anisotropic expansion factor**2
                                    tmp.kelem(idx) *= std::pow(lss_aniso_alpha[idim],2.0);
                                }

                                // divide by Lbox, because displacement is in box units for output plugin
                                tmp.kelem(idx) *= vunit / boxlen;
                            }
                        }
                    }
                    tmp.zero_DC_mode();
                    tmp.FourierTransformBackward();

                    // if we write particle data, store particle data in particle structure
                    if( the_output_plugin->write_species_as( this_species ) == output_type::particles )
                    {
                        particle_lattice_generator_ptr->set_velocities( lattice_type, shifted_lattice, idim, the_output_plugin->has_64bit_reals(), tmp, the_config );
                    }
                    // otherwise write out the grid data directly to the output plugin
                    else if( the_output_plugin->write_species_as( this_species ) == output_type::field_lagrangian )
                    {
                        fluid_component fc = (idim==0)? fluid_component::vx : ((idim==1)? fluid_component::vy : fluid_component::vz );
                        the_output_plugin->write_grid_data( tmp, this_species, fc );
                    }
                }

                if( the_output_plugin->write_species_as( this_species ) == output_type::particles )
                {
                    the_output_plugin->write_particle_data( particle_lattice_generator_ptr->get_particles(), this_species, Omega[this_species] );
                }
                
                if( the_output_plugin->write_species_as( this_species ) == output_type::field_lagrangian )
                {
                    // use density simply from 1st order SPT
                    phi.FourierTransformForward();
                    tmp.FourierTransformForward(false);
                    tmp.assign_function_of_grids_kdep( []( auto kvec, auto pphi ){
                        return - kvec.norm_squared() *  pphi;
                    }, phi);
                    tmp.Write_PowerSpectrum("input_powerspec_sampled_SPT.txt");
                    tmp.FourierTransformBackward();
                    the_output_plugin->write_grid_data( tmp, this_species, fluid_component::density );
                }
            }

        }
        
        if( the_output_plugin->write_species_as( this_species ) == output_type::particles
            && this_species == cosmo_species::neutrino)
        {
#ifdef USE_FASTDF
            struct params pars;
            struct units us;

            // Set up FastDF unit system
            us.UnitLengthMetres = 3.085677581491e+022; // Mpc
            us.UnitTimeSeconds = 3.08567758148957E+019; // for km/s velocities
            us.UnitMassKilogram = 1.988435e40; // 1e10 M_sol
            us.UnitTemperatureKelvin = 1.0;
            us.UnitCurrentAmpere = 1.0;

            setPhysicalConstants(&us);

            // Set up FastDF parameters
            initParams(&pars);
            pars.FirstID = ngrid * ngrid * ngrid + 1; // immediately after the dm particles
            pars.CubeRootNumber = NeutrinoCubeRootNum;
            pars.NumPartGenerate = TotalNeutrinoNum;
            pars.ScaleFactorBegin = 1e-9;
            pars.ScaleFactorEnd = 1.0/(zstart + 1.0);
            pars.ScaleFactorStep = NeutrinoStepSize;
            pars.RecomputeTrigger = 0.01;
            pars.RecomputeScaleRef = 0.0;
            pars.InterpolationOrder = NeutrinoInterpOrder;
            pars.InvertField = 0;
            pars.BoxLen = boxlen / the_cosmo_calc->cosmo_param_["h"];
            pars.AlternativeEquations = 0;
            pars.OutputFields = 0;
            pars.NormalizeGaussianField = 0;
            pars.AssumeMonofonicNormalization = 1;
            pars.PrimordialScalarAmplitude = the_cosmo_calc->cosmo_param_["A_s"];
            pars.PrimordialSpectralIndex = the_cosmo_calc->cosmo_param_["n_s"];
            pars.PrimordialPivotScale = the_cosmo_calc->cosmo_param_["k_p"];
            pars.PrimordialRunning = the_cosmo_calc->cosmo_param_["alpha_s"];
            pars.PrimordialRunningSecond = the_cosmo_calc->cosmo_param_["beta_s"];

            std::string out_fname;
            out_fname = the_config.get_value<std::string>("output", "filename");

            std::string nupart_density_tfunc = "d_ncdm[0]";
            std::string gauge = "N-body";
            std::string class_parameter_file = "input_class_parameters.ini";

            // Determine output settings depending on the output plugin
            std::string out_plug = the_config.get_value<std::string>("output", "format");
            std::string velocity_type;
            int include_h_factor;
            int distributed_files;
            if (out_plug == "gadget_hdf5" || out_plug == "AREPO") {
                velocity_type = "Gadget";
                include_h_factor = 1;
                distributed_files = 1;
            } else if (out_plug == "SWIFT") {
                velocity_type = "peculiar";
                include_h_factor = 0;
                distributed_files = 0;
            } else {
                throw std::runtime_error("Output format not supported by FastDF (only HDF5 particle formats, e.g. Gadget, SWIFT).");
            }

            // The group name for this species, according to the output plugin
            int export_key = the_output_plugin->get_species_idx(this_species);
            std::string export_name = std::string("PartType") + std::to_string(export_key);

            // Should coordinates and masses include a factor of h^-1?
            pars.IncludeHubbleFactors = include_h_factor;
            // Does each task create its own file?
            pars.DistributedFiles = distributed_files;

            // Set string parameters
            strcpy(pars.OutputDirectory, ".");
            strcpy(pars.ExportName, export_name.c_str());
            strcpy(pars.OutputFilename, out_fname.c_str());
            strcpy(pars.GaussianRandomFieldFile, white_noise_fname.c_str());
            strcpy(pars.GaussianRandomFieldDataset, white_noise_dset.c_str());
            strcpy(pars.TransferFunctionDensity, nupart_density_tfunc.c_str());
            strcpy(pars.Gauge, gauge.c_str());
            strcpy(pars.ClassIniFile, class_parameter_file.c_str());
            strcpy(pars.VelocityType, velocity_type.c_str());

            uint64_t num_local_nuparts = run_fastdf(&pars, &us);

            music::ilog << std::endl;

            // Also set Header attributes
            the_output_plugin->set_particle_attributes(num_local_nuparts, (uint64_t) (TotalNeutrinoNum), this_species, the_cosmo_calc->cosmo_param_["Omega_nu_1"]);
#else
            throw std::runtime_error("Not compiled with FastDF.");
#endif
        }

        music::ilog << "-------------------------------------------------------------------------------" << std::endl;
        
    }
    return 0;
}


} // end namespace ic_generator

