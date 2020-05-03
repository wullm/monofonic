
#include <general.hh>
#include <grid_fft.hh>
#include <operators.hh>
#include <convolution.hh>
#include <testing.hh>

#include <ic_generator.hh>
#include <particle_generator.hh>
#include <particle_plt.hh>

#include <unistd.h> // for unlink

std::map<cosmo_species,std::string> cosmo_species_name = 
{
  {cosmo_species::dm,"Dark matter"},
  {cosmo_species::baryon,"Baryons"},
  {cosmo_species::neutrino,"Neutrinos"}
};

namespace ic_generator{

std::unique_ptr<RNG_plugin> the_random_number_generator;
std::unique_ptr<output_plugin> the_output_plugin;
std::unique_ptr<cosmology::calculator>  the_cosmo_calc;

int Initialise( config_file& the_config )
{
    the_random_number_generator = std::move(select_RNG_plugin(the_config));
    the_output_plugin           = std::move(select_output_plugin(the_config));
    the_cosmo_calc              = std::make_unique<cosmology::calculator>(the_config);

    return 0;
}

int Run( config_file& the_config )
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
    int LPTorder = the_config.get_value_safe<double>("setup","LPTorder",100);

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
    const bool bDoFixing = the_config.get_value_safe<bool>("setup", "DoFixing", false);

    //--------------------------------------------------------------------------------------------------------
    //! do baryon ICs?
    const bool bDoBaryons = the_config.get_value_safe<bool>("setup", "DoBaryons", false );
    std::map< cosmo_species, double > Omega;
    if( bDoBaryons ){
        double Om = the_config.get_value<double>("cosmology", "Omega_m");
        double Ob = the_config.get_value<double>("cosmology", "Omega_b");
        Omega[cosmo_species::dm] = Om-Ob;
        Omega[cosmo_species::baryon] = Ob;
    }else{
        double Om = the_config.get_value<double>("cosmology", "Omega_m");
        Omega[cosmo_species::dm] = Om;
        Omega[cosmo_species::baryon] = 0.0;
    }

    //--------------------------------------------------------------------------------------------------------
    //! do constrained ICs?
    const bool bAddConstrainedModes =  the_config.contains_key("setup", "ConstraintFieldFile" );

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
    // Anisotropy parameters for beyond box tidal field 
    std::array<real_t,3> lss_aniso_lambda = {
        the_config.get_value_safe<double>("cosmology", "LSS_aniso_lx", 0.0),
        the_config.get_value_safe<double>("cosmology", "LSS_aniso_ly", 0.0),
        the_config.get_value_safe<double>("cosmology", "LSS_aniso_lz", 0.0),
    };  
    
    if( std::abs(lss_aniso_lambda[0]+lss_aniso_lambda[1]+lss_aniso_lambda[2]) > 1e-10 ){
        music::elog << "External tidal field is not trace-free! Will subtract trace!" << std::endl;
        auto tr_l_3 = (lss_aniso_lambda[0]+lss_aniso_lambda[1]+lss_aniso_lambda[2])/3.0;
        lss_aniso_lambda[0] -= tr_l_3;
        lss_aniso_lambda[1] -= tr_l_3;
        lss_aniso_lambda[2] -= tr_l_3;
    }

    //--------------------------------------------------------------------------------------------------------

    const real_t astart = 1.0/(1.0+zstart);
    const real_t volfac(std::pow(boxlen / ngrid / 2.0 / M_PI, 1.5));

    the_cosmo_calc->write_powerspectrum(astart, "input_powerspec.txt" );

    //music::ilog << "-----------------------------------------------------------------------------" << std::endl;

    // if( bSymplecticPT && LPTorder!=2 ){
    //     music::wlog << "SymplecticPT has been selected and will overwrite chosen order of LPT to 2" << std::endl;
    //     LPTorder = 2;
    // }

    //--------------------------------------------------------------------
    // Compute LPT time coefficients
    //--------------------------------------------------------------------
    const real_t Dplus0 = the_cosmo_calc->get_growth_factor(astart);
    const real_t vfac   = the_cosmo_calc->get_vfact(astart);

    const double g1  = -Dplus0;
    const double g2  = ((LPTorder>1)? -3.0/7.0*Dplus0*Dplus0 : 0.0);
    const double g3a = ((LPTorder>2)? 1.0/3.0*Dplus0*Dplus0*Dplus0 : 0.0);
    const double g3b = ((LPTorder>2)? -10.0/21.*Dplus0*Dplus0*Dplus0 : 0.0);
    const double g3c = ((LPTorder>2)? 1.0/7.0*Dplus0*Dplus0*Dplus0 : 0.0);

    // vfac = d log D+ / dt 
    // d(D+^2)/dt = 2*D+ * d D+/dt = 2 * D+^2 * vfac
    // d(D+^3)/dt = 3*D+^2* d D+/dt = 3 * D+^3 * vfac
    const double vfac1 =  vfac;
    const double vfac2 =  2*vfac;
    const double vfac3 =  3*vfac;

    // coefficients needed for anisotropic external tides
    const double ai3 = std::pow(astart,-3);
    const double Omega_m_of_a = the_cosmo_calc->cosmo_param_.Omega_m * ai3 / (the_cosmo_calc->cosmo_param_.Omega_m * ai3 + the_cosmo_calc->cosmo_param_.Omega_DE);
    const double f1 = the_cosmo_calc->get_f(astart);
    const double f_aniso = -4.0/3.0 * f1 * f1 / Omega_m_of_a;

    const std::array<real_t,3> lss_aniso_alpha = {
        1.0 - Dplus0 * lss_aniso_lambda[0],
        1.0 - Dplus0 * lss_aniso_lambda[1],
        1.0 - Dplus0 * lss_aniso_lambda[2],
    };

    //--------------------------------------------------------------------
    // Create arrays
    //--------------------------------------------------------------------
    Grid_FFT<real_t> phi({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> phi2({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> phi3a({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> phi3b({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> A3x({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> A3y({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> A3z({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    //... array [.] access to components of A3:
    std::array<Grid_FFT<real_t> *, 3> A3({&A3x, &A3y, &A3z});

    // white noise field 
    Grid_FFT<real_t> wnoise({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    
    //--------------------------------------------------------------------
    // Fill the grid with a Gaussian white noise field
    //--------------------------------------------------------------------
    music::ilog << "-------------------------------------------------------------------------------" << std::endl;
    music::ilog << "Generating white noise field...." << std::endl;

    the_random_number_generator->Fill_Grid(wnoise);
    
    wnoise.FourierTransformForward();

    //--------------------------------------------------------------------
    // Use externally specified large scale modes from constraints in case
    //--------------------------------------------------------------------
    if( bAddConstrainedModes ){
        Grid_FFT<real_t,false> cwnoise({8,8,8}, {boxlen,boxlen,boxlen});
        cwnoise.Read_from_HDF5( the_config.get_value<std::string>("setup", "ConstraintFieldFile"), 
                the_config.get_value<std::string>("setup", "ConstraintFieldName") );
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
            if( i>ngrid_c_2 && i+ngrid-ngrid_c>ngrid/2) il = ngrid-ngrid_c+i;
            if( il == size_t(-1) ) continue;
            if( il<size_t(wnoise.local_1_start_) || il>=size_t(wnoise.local_1_start_+wnoise.local_1_size_)) continue;
            il -= wnoise.local_1_start_;
            for( size_t j=0; j<ngrid_c; ++j ){
                size_t jl = size_t(-1);
                if( j<ngrid_c_2 && j<ngrid/2 ) jl = j;
                if( j>ngrid_c_2 && j+ngrid-ngrid_c>ngrid/2 ) jl = ngrid-ngrid_c+j;
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
        if (bDoFixing)
            wn = (std::abs(wn) != 0.0) ? wn / std::abs(wn) : wn;
        return wn / volfac;
    });


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

    //======================================================================
    //... compute 1LPT displacement potential ....
    //======================================================================
    // phi = - delta / k^2

    music::ilog << "-------------------------------------------------------------------------------" << std::endl;
    music::ilog << "Generating white noise field...." << std::endl;

    double wtime = get_wtime();
    music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(1) term" << std::flush;

    phi.FourierTransformForward(false);
    phi.assign_function_of_grids_kdep([&](auto k, auto wn) {
        real_t kmod = k.norm();
        ccomplex_t delta = wn * the_cosmo_calc->get_amplitude(kmod, total);
        return -delta / (kmod * kmod);
    }, wnoise);

    phi.zero_DC_mode();

    music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime() - wtime << "s" << std::endl;

    //======================================================================
    //... compute 2LPT displacement potential ....
    //======================================================================
    if (LPTorder > 1)
    {
        wtime = get_wtime();
        music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(2) term" << std::flush;
        phi2.FourierTransformForward(false);
        Conv.convolve_SumOfHessians(phi, {0, 0}, phi, {1, 1}, {2, 2}, op::assign_to(phi2));
        Conv.convolve_Hessians(phi, {1, 1}, phi, {2, 2}, op::add_to(phi2));
        Conv.convolve_Hessians(phi, {0, 1}, phi, {0, 1}, op::subtract_from(phi2));
        Conv.convolve_Hessians(phi, {0, 2}, phi, {0, 2}, op::subtract_from(phi2));
        Conv.convolve_Hessians(phi, {1, 2}, phi, {1, 2}, op::subtract_from(phi2));

        if (bAddExternalTides)
        {
            phi2.assign_function_of_grids_kdep([&](vec3_t<real_t> kvec, ccomplex_t pphi, ccomplex_t pphi2) {
                // sign in front of f_aniso is reversed since phi1 = -phi
                return pphi2 + f_aniso * (kvec[0] * kvec[0] * lss_aniso_lambda[0] + kvec[1] * kvec[1] * lss_aniso_lambda[1] + kvec[2] * kvec[2] * lss_aniso_lambda[2]) * pphi;
            },
                                               phi, phi2);
        }

        phi2.apply_InverseLaplacian();
        music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime() - wtime << "s" << std::endl;

        if (bAddExternalTides)
        {
            music::wlog << "Added external tide contribution to phi(2)... Make sure your N-body code supports this!" << std::endl;
            music::wlog << " lss_aniso = (" << lss_aniso_lambda[0] << ", " << lss_aniso_lambda[1] << ", " << lss_aniso_lambda[2] << ")" << std::endl;
        }
    }

    //======================================================================
    //... compute 3LPT displacement potential
    //======================================================================
    if (LPTorder > 2)
    {
        //... 3a term ...
        wtime = get_wtime();
        music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(3a) term" << std::flush;
        phi3a.FourierTransformForward(false);
        Conv.convolve_Hessians(phi, {0, 0}, phi, {1, 1}, phi, {2, 2}, op::assign_to(phi3a));
        Conv.convolve_Hessians(phi, {0, 1}, phi, {0, 2}, phi, {1, 2}, op::multiply_add_to(phi3a,2.0));
        Conv.convolve_Hessians(phi, {1, 2}, phi, {1, 2}, phi, {0, 0}, op::subtract_from(phi3a));
        Conv.convolve_Hessians(phi, {0, 2}, phi, {0, 2}, phi, {1, 1}, op::subtract_from(phi3a));
        Conv.convolve_Hessians(phi, {0, 1}, phi, {0, 1}, phi, {2, 2}, op::subtract_from(phi3a));
        phi3a.apply_InverseLaplacian();
        music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime() - wtime << "s" << std::endl;

        //... 3b term ...
        wtime = get_wtime();
        music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(3b) term" << std::flush;
        phi3b.FourierTransformForward(false);
        Conv.convolve_SumOfHessians(phi, {0, 0}, phi2, {1, 1}, {2, 2}, op::assign_to(phi3b));
        Conv.convolve_SumOfHessians(phi, {1, 1}, phi2, {2, 2}, {0, 0}, op::add_to(phi3b));
        Conv.convolve_SumOfHessians(phi, {2, 2}, phi2, {0, 0}, {1, 1}, op::add_to(phi3b));
        Conv.convolve_Hessians(phi, {0, 1}, phi2, {0, 1}, op::multiply_add_to(phi3b,-2.0));
        Conv.convolve_Hessians(phi, {0, 2}, phi2, {0, 2}, op::multiply_add_to(phi3b,-2.0));
        Conv.convolve_Hessians(phi, {1, 2}, phi2, {1, 2}, op::multiply_add_to(phi3b,-2.0));
        phi3b.apply_InverseLaplacian();
        phi3b *= 0.5; // factor 1/2 from definition of phi(3b)!
        music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime() - wtime << "s" << std::endl;

        //... transversal term ...
        wtime = get_wtime();
        music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing A(3) term" << std::flush;
        for (int idim = 0; idim < 3; ++idim)
        {
            // cyclic rotations of indices
            int idimp = (idim + 1) % 3, idimpp = (idim + 2) % 3;
            A3[idim]->FourierTransformForward(false);
            Conv.convolve_Hessians(phi2, {idim, idimp}, phi, {idim, idimpp}, op::assign_to(*A3[idim]));
            Conv.convolve_Hessians(phi2, {idim, idimpp}, phi, {idim, idimp}, op::subtract_from(*A3[idim]));
            Conv.convolve_DifferenceOfHessians(phi, {idimp, idimpp}, phi2, {idimp, idimp}, {idimpp, idimpp}, op::add_to(*A3[idim]));
            Conv.convolve_DifferenceOfHessians(phi2, {idimp, idimpp}, phi, {idimp, idimp}, {idimpp, idimpp}, op::subtract_from(*A3[idim]));
            A3[idim]->apply_InverseLaplacian();
        }
        music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime() - wtime << "s" << std::endl;
    }

    // if( bSymplecticPT ){
    //     //... transversal term ...
    //     wtime = get_wtime();
    //     music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing vNLO(3) term" << std::flush;
    //     for( int idim=0; idim<3; ++idim ){
    //         // cyclic rotations of indices
    //         A3[idim]->FourierTransformForward(false);
    //         Conv.convolve_Gradient_and_Hessian( phi, {0},  phi2, {idim,0}, assign_to(*A3[idim]) );
    //         Conv.convolve_Gradient_and_Hessian( phi, {1},  phi2, {idim,1}, add_to(*A3[idim]) );
    //         Conv.convolve_Gradient_and_Hessian( phi, {2},  phi2, {idim,2}, add_to(*A3[idim]) );
    //     }
    //     music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime()-wtime << "s" << std::endl;

    // }

    ///... scale all potentials with respective growth factors
    phi *= g1;
    phi2 *= g2;
    phi3a *= g3a;
    phi3b *= g3b;
    (*A3[0]) *= g3c;
    (*A3[1]) *= g3c;
    (*A3[2]) *= g3c;

    music::ilog << "-------------------------------------------------------------------------------" << std::endl;

    ///////////////////////////////////////////////////////////////////////
    // we store the densities here if we compute them
    //======================================================================

    // Testing
    const std::string testing = the_config.get_value_safe<std::string>("testing", "test", "none");

    if (testing != "none")
    {
        music::wlog << "you are running in testing mode. No ICs, only diagnostic output will be written out!" << std::endl;
        if (testing == "potentials_and_densities"){
            testing::output_potentials_and_densities(the_config, ngrid, boxlen, phi, phi2, phi3a, phi3b, A3);
        }
        else if (testing == "velocity_displacement_symmetries"){
            testing::output_velocity_displacement_symmetries(the_config, ngrid, boxlen, vfac, Dplus0, phi, phi2, phi3a, phi3b, A3);
        }
        else if (testing == "convergence"){
            testing::output_convergence(the_config, the_cosmo_calc.get(), ngrid, boxlen, vfac, Dplus0, phi, phi2, phi3a, phi3b, A3);
        }
        else{
            music::flog << "unknown test '" << testing << "'" << std::endl;
            std::abort();
        }
    }

    for( auto& this_species : species_list )
    {
        music::ilog << std::endl
                    << ">>> Computing ICs for species \'" << cosmo_species_name[this_species] << "\' <<<\n" << std::endl;

        {
            // temporary storage of data
            Grid_FFT<real_t> tmp({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});


            //if( the_output_plugin->write_species_as( cosmo_species::dm ) == output_type::field_eulerian ){
            if( the_output_plugin->write_species_as(this_species) == output_type::field_eulerian )
            {
                //======================================================================
                // use QPT to get density and velocity fields
                //======================================================================
                Grid_FFT<ccomplex_t> psi({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
                Grid_FFT<real_t> rho({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

                //======================================================================
                // initialise psi = exp(i Phi(1)/hbar)
                //======================================================================
                phi.FourierTransformBackward();
                real_t std_phi1 = phi.std();

                const real_t hbar = 2.0 * M_PI/ngrid * (2*std_phi1/Dplus0); //3sigma, but this might rather depend on gradients of phi...
                music::ilog << "Semiclassical PT : hbar = " << hbar << " from sigma(phi1) = " << std_phi1 << std::endl;
                
                if( LPTorder == 1 ){
                    psi.assign_function_of_grids_r([hbar,Dplus0]( real_t pphi ){
                        return std::exp(ccomplex_t(0.0,1.0/hbar) * (pphi / Dplus0));
                    }, phi );
                }else if( LPTorder >= 2 ){
                    phi2.FourierTransformBackward();
                    // we don't have a 1/2 in the Veff term because pre-factor is already 3/7
                    psi.assign_function_of_grids_r([hbar,Dplus0]( real_t pphi, real_t pphi2 ){
                        return std::exp(ccomplex_t(0.0,1.0/hbar) * (pphi + pphi2) / Dplus0);
                    }, phi, phi2 );
                    // phi2.FourierTransformBackward();
                }
                // phi.FourierTransformForward();

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
                for( int idim=0; idim<3; ++idim )
                {
                    Grid_FFT<ccomplex_t> grad_psi({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
                    grad_psi.copy_from(psi);
                    grad_psi.FourierTransformForward();
                    grad_psi.apply_function_k_dep([&](auto x, auto k) {
                        return x * ccomplex_t(0.0,k[idim]);
                    });
                    grad_psi.FourierTransformBackward();
                    psi.FourierTransformBackward();

                    tmp.assign_function_of_grids_r([&](auto ppsi, auto pgrad_psi, auto prho) {
                            return std::real((std::conj(ppsi) * pgrad_psi - ppsi * std::conj(pgrad_psi)) / ccomplex_t(0.0, 2.0 / hbar)/(1.0+prho));
                        }, psi, grad_psi, rho);

                    fluid_component fc = (idim==0)? fluid_component::vx : ((idim==1)? fluid_component::vy : fluid_component::vz );
                    the_output_plugin->write_grid_data( tmp, this_species, fc );
                }
            }

            if( the_output_plugin->write_species_as( this_species ) == output_type::particles 
             || the_output_plugin->write_species_as( this_species ) == output_type::field_lagrangian )
            {
                //===================================================================================
                // we store displacements and velocities here if we compute them
                //===================================================================================
                particle::container particles;

                bool shifted_lattice = (this_species == cosmo_species::baryon &&
                                        the_output_plugin->write_species_as(this_species) == output_type::particles) ? true : false;

                // somewhat arbitrarily, start baryon particle IDs from 2**31 if we have 32bit and from 2**56 if we have 64 bits
                size_t IDoffset = (this_species == cosmo_species::baryon)? ((the_output_plugin->has_64bit_ids())? 1ul<<56 : 1ul<<31): 0 ;

                grid_interpolate<1,Grid_FFT<real_t>> interp( tmp );

                // if output plugin wants particles, then we need to store them, along with their IDs
                if( the_output_plugin->write_species_as( this_species ) == output_type::particles )
                {
                    // allocate particle structure and generate particle IDs
                    particle::initialize_lattice( particles, lattice_type, the_output_plugin->has_64bit_reals(), the_output_plugin->has_64bit_ids(), IDoffset, tmp, the_config );
                }
            
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
                                auto phitot = phi.kelem(idx) + phi2.kelem(idx) + phi3a.kelem(idx) + phi3b.kelem(idx);
                                // divide by Lbox, because displacement is in box units for output plugin
                                tmp.kelem(idx) = lunit / boxlen * ( lg.gradient(idim,tmp.get_k3(i,j,k)) * phitot 
                                    + lg.gradient(idimp,tmp.get_k3(i,j,k)) * A3[idimpp]->kelem(idx) - lg.gradient(idimpp,tmp.get_k3(i,j,k)) * A3[idimp]->kelem(idx) );

                                if( the_output_plugin->write_species_as( this_species ) == output_type::particles && lattice_type == particle::lattice_glass){
                                    tmp.kelem(idx) *= interp.compensation_kernel( tmp.get_k<real_t>(i,j,k) );
                                }

                                if( bDoBaryons ){
                                    vec3_t<real_t> kvec = phi.get_k<real_t>(i,j,k);
                                    real_t k2 = kvec.norm_squared(), kmod = std::sqrt(k2);
                                    // double ampldiff = ((this_species == cosmo_species::dm)? the_cosmo_calc->get_amplitude(kmod, cdm) :
                                    //  (this_species == cosmo_species::baryon)? the_cosmo_calc->get_amplitude(kmod, baryon) : 
                                    // //   the_cosmo_calc->get_amplitude(kmod, total)) - the_cosmo_calc->get_amplitude(kmod, total);
                                    //  the_cosmo_calc->get_amplitude(kmod, total)*(-g1)) - the_cosmo_calc->get_amplitude(kmod, total)*(-g1);

                                    real_t ampldiff = (((this_species == cosmo_species::dm)? the_cosmo_calc->get_amplitude(kmod, cdm) 
                                        : (this_species == cosmo_species::baryon)? the_cosmo_calc->get_amplitude(kmod, baryon) : 
                                           the_cosmo_calc->get_amplitude(kmod, total)) - the_cosmo_calc->get_amplitude(kmod, total)) * (-g1);

                                    tmp.kelem(idx) += lg.gradient(idim, tmp.get_k3(i,j,k)) * wnoise.kelem(idx) * lunit * ampldiff / k2 / boxlen;
                                }
                            }
                        }
                    }
                    tmp.zero_DC_mode();
                    tmp.FourierTransformBackward();

                    // if we write particle data, store particle data in particle structure
                    if( the_output_plugin->write_species_as( this_species ) == output_type::particles )
                    {
                        particle::set_positions( particles, lattice_type, shifted_lattice, idim, lunit, the_output_plugin->has_64bit_reals(), tmp, the_config );
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
                                // divide by Lbox, because displacement is in box units for output plugin
                                auto phitot_v = vfac1 * phi.kelem(idx) + vfac2 * phi2.kelem(idx) + vfac3 * (phi3a.kelem(idx) + phi3b.kelem(idx));

                                tmp.kelem(idx) = vunit / boxlen * ( lg.gradient(idim,tmp.get_k3(i,j,k)) * phitot_v 
                                        + vfac3 * (lg.gradient(idimp,tmp.get_k3(i,j,k)) * A3[idimpp]->kelem(idx) - lg.gradient(idimpp,tmp.get_k3(i,j,k)) * A3[idimp]->kelem(idx)) );

                                if( the_output_plugin->write_species_as( this_species ) == output_type::particles && lattice_type == particle::lattice_glass){
                                    tmp.kelem(idx) *= interp.compensation_kernel( tmp.get_k<real_t>(i,j,k) );
                                }

                                if( bDoBaryons ){
                                    vec3_t<real_t> kvec = phi.get_k<real_t>(i,j,k);
                                    real_t k2 = kvec.norm_squared(), kmod = std::sqrt(k2);
                                    // double ampldiff = ((this_species == cosmo_species::dm)? the_cosmo_calc->get_amplitude(kmod, vcdm0) :
                                    //  (this_species == cosmo_species::baryon)? the_cosmo_calc->get_amplitude(kmod, vbaryon0) : 
                                    //      the_cosmo_calc->get_amplitude(kmod, vtotal0)) - the_cosmo_calc->get_amplitude(kmod, vtotal0);
                                    // // the_cosmo_calc->get_amplitude(kmod, total)*(-g1)) - the_cosmo_calc->get_amplitude(kmod, total)*(-g1);
                                    real_t ampldiff = (((this_species == cosmo_species::dm)? the_cosmo_calc->get_amplitude(kmod, vcdm) 
                                        : (this_species == cosmo_species::baryon)? the_cosmo_calc->get_amplitude(kmod, vbaryon) : 
                                           the_cosmo_calc->get_amplitude(kmod, vtotal)) - the_cosmo_calc->get_amplitude(kmod, vtotal)) * (-g1);
                                    tmp.kelem(idx) += lg.gradient(idim, tmp.get_k3(i,j,k)) * wnoise.kelem(idx) * vfac1 * vunit / boxlen * ampldiff / k2 ;
                                }

                                // correct velocity with PLT mode growth rate
                                tmp.kelem(idx) *= lg.vfac_corr(tmp.get_k3(i,j,k));


                                if( bAddExternalTides ){
                                    // modify velocities with anisotropic expansion factor**2
                                    tmp.kelem(idx) *= std::pow(lss_aniso_alpha[idim],2.0);
                                }
                                // if( bSymplecticPT){
                                //     auto phitot_v = vfac1 * phi.kelem(idx) + vfac2 * phi2.kelem(idx);
                                //     tmp.kelem(idx) = vunit*ccomplex_t(0.0,1.0) * (kk[idim] * phitot_v) + vfac1 * A3[idim]->kelem(idx);
                                // }
                            }
                        }
                    }
                    tmp.zero_DC_mode();
                    tmp.FourierTransformBackward();

                    // if we write particle data, store particle data in particle structure
                    if( the_output_plugin->write_species_as( this_species ) == output_type::particles )
                    {
                        particle::set_velocities( particles, lattice_type, shifted_lattice, idim, the_output_plugin->has_64bit_reals(), tmp, the_config );
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
                    the_output_plugin->write_particle_data( particles, this_species, Omega[this_species] );
                }
                
                if( the_output_plugin->write_species_as( this_species ) == output_type::field_lagrangian )
                {
                    // use density simply from 1st order SPT
                    phi.FourierTransformForward();
                    phi.apply_negative_Laplacian();
                    phi.Write_PowerSpectrum("input_powerspec_sampled_SPT.txt");
                    phi.FourierTransformBackward();
                    the_output_plugin->write_grid_data( phi, this_species, fluid_component::density );
                }
            }

        }
    }
    return 0;
}


} // end namespace ic_generator

