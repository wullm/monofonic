
#include <general.hh>
#include <grid_fft.hh>
#include <convolution.hh>

#include <ic_generator.hh>

#include <unistd.h> // for unlink

namespace ic_generator{

std::unique_ptr<RNG_plugin> the_random_number_generator;
std::unique_ptr<output_plugin> the_output_plugin;
std::unique_ptr<CosmologyCalculator>  the_cosmo_calc;

int Initialise( ConfigFile& the_config )
{
    the_random_number_generator = std::move(select_RNG_plugin(the_config));
    the_output_plugin           = std::move(select_output_plugin(the_config));
    the_cosmo_calc              = std::make_unique<CosmologyCalculator>(the_config);

    return 0;
}

int Run( ConfigFile& the_config )
{
    //--------------------------------------------------------------------
    // Read run parameters
    //--------------------------------------------------------------------

    const size_t ngrid = the_config.GetValue<size_t>("setup", "GridRes");
    const real_t boxlen = the_config.GetValue<double>("setup", "BoxLength");
    const real_t zstart = the_config.GetValue<double>("setup", "zstart");
    int LPTorder = the_config.GetValueSafe<double>("setup","LPTorder",100);
    const bool initial_bcc_lattice = the_config.GetValueSafe<bool>("setup","BCClattice",false);
    const bool bSymplecticPT = the_config.GetValueSafe<bool>("setup","SymplecticPT",false);
    const real_t astart = 1.0/(1.0+zstart);
    const real_t volfac(std::pow(boxlen / ngrid / 2.0 / M_PI, 1.5));

    const bool bDoFixing = the_config.GetValueSafe<bool>("setup", "DoFixing",false);
    
    the_cosmo_calc->WritePowerspectrum(astart, "input_powerspec.txt" );

    csoca::ilog << "-----------------------------------------------------------------------------" << std::endl;

    if( bSymplecticPT && LPTorder!=2 ){
        csoca::wlog << "SymplecticPT has been selected and will overwrite chosen order of LPT to 2" << std::endl;
        LPTorder = 2;
    }

    //--------------------------------------------------------------------
    // Compute LPT time coefficients
    //--------------------------------------------------------------------
    const real_t Dplus0 = the_cosmo_calc->CalcGrowthFactor(astart) / the_cosmo_calc->CalcGrowthFactor(1.0);
    const real_t vfac   = the_cosmo_calc->CalcVFact(astart);

    const double g1  = -Dplus0;
    const double g2  = ((LPTorder>1)? -3.0/7.0*Dplus0*Dplus0 : 0.0);
    const double g3a = ((LPTorder>2)? -1.0/3.0*Dplus0*Dplus0*Dplus0 : 0.0);
    const double g3b = ((LPTorder>2)? 10.0/21.*Dplus0*Dplus0*Dplus0 : 0.0);
    const double g3c = ((LPTorder>2)? -1.0/7.0*Dplus0*Dplus0*Dplus0 : 0.0);

    const double vfac1 =  vfac;
    const double vfac2 =  2*vfac1;
    const double vfac3 =  3*vfac1;

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
    std::array< Grid_FFT<real_t>*,3 > A3({&A3x,&A3y,&A3z});

    //--------------------------------------------------------------------
    // Create convolution class instance for non-linear terms
    //--------------------------------------------------------------------
    OrszagConvolver<real_t> Conv({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    // NaiveConvolver<real_t> Conv({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    //--------------------------------------------------------------------
    // Some operators to add or subtract terms 
    auto assign_to = [](auto &g){return [&](auto i, auto v){ g[i] = v; };};
    auto add_to = [](auto &g){return [&](auto i, auto v){ g[i] += v; };};
    auto add_twice_to = [](auto &g){return [&](auto i, auto v){ g[i] += 2*v; };};
    auto subtract_from = [](auto &g){return [&](auto i, auto v){ g[i] -= v; };};
    auto subtract_twice_from = [](auto &g){return [&](auto i, auto v){ g[i] -= 2*v; };};
    //--------------------------------------------------------------------
    
    
    //======================================================================
    //... compute 1LPT displacement potential ....
    //======================================================================
    // phi = - delta / k^2
    double wtime = get_wtime();    
    csoca::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(1) term" << std::flush;

    #if 1 //  random ICs
    //--------------------------------------------------------------------
    // Fill the grid with a Gaussian white noise field
    //--------------------------------------------------------------------
    the_random_number_generator->Fill_Grid( phi );

    phi.FourierTransformForward();
    
    phi.apply_function_k_dep([&](auto x, auto k) -> ccomplex_t {
        real_t kmod = k.norm();
        if( bDoFixing ) x = (std::abs(x)!=0.0)? x / std::abs(x) : x; 
        ccomplex_t delta = x * the_cosmo_calc->GetAmplitude(kmod, total);
        return -delta / (kmod * kmod) / volfac;
    });

    phi.zero_DC_mode();
    #else // ICs with a given phi(1) potential function
    constexpr real_t twopi{2.0*M_PI};
    constexpr real_t epsilon_q1d{0.25};

    constexpr real_t epsy{0.25};
    constexpr real_t epsz{0.25};
    
    phi.FourierTransformBackward(false);

    phi.apply_function_r_dep([&](auto v, auto r) -> real_t {
        real_t q1 = r[0]-0.5*boxlen;//r[0]/boxlen * twopi - M_PI;
        real_t q2 = r[1]-0.5*boxlen;//r[1]/boxlen * twopi - M_PI;
        real_t q3 = r[2]-0.5*boxlen;//r[1]/boxlen * twopi - M_PI;

        // std::cerr << q1  << " " << q2 << std::endl;
        
        return -2.0*std::cos(q1+std::cos(q2));
        // return (-std::cos(q1) + epsilon_q1d * std::sin(q2));
        // return (-std::cos(q1) + epsy * std::sin(q2) + epsz * std::cos(q1) * std::sin(q3));
    });
    phi.FourierTransformForward();


    #endif
    csoca::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime()-wtime << "s" << std::endl;
    
    //======================================================================
    //... compute 2LPT displacement potential ....
    //======================================================================
    if( LPTorder > 1 || bSymplecticPT ){
        wtime = get_wtime();    
        csoca::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(2) term" << std::flush;
        phi2.FourierTransformForward(false);
        Conv.convolve_SumOfHessians( phi, {0,0}, phi, {1,1}, {2,2}, assign_to( phi2 ) );
        Conv.convolve_Hessians( phi, {1,1}, phi, {2,2}, add_to(phi2) );
        Conv.convolve_Hessians( phi, {0,1}, phi, {0,1}, subtract_from(phi2) );
        Conv.convolve_Hessians( phi, {0,2}, phi, {0,2}, subtract_from(phi2) );
        Conv.convolve_Hessians( phi, {1,2}, phi, {1,2}, subtract_from(phi2) );
        phi2.apply_InverseLaplacian();
        csoca::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime()-wtime << "s" << std::endl;
    }

    //======================================================================
    //... compute 3LPT displacement potential
    //======================================================================
    if( LPTorder > 2  && !bSymplecticPT ){
        //... 3a term ...
        wtime = get_wtime();    
        csoca::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(3a) term" << std::flush;
        phi3a.FourierTransformForward(false);
        Conv.convolve_Hessians( phi, {0,0}, phi, {1,1}, phi, {2,2}, assign_to(phi3a) );
        Conv.convolve_Hessians( phi, {0,1}, phi, {0,2}, phi, {1,2}, add_twice_to(phi3a) );
        Conv.convolve_Hessians( phi, {1,2}, phi, {1,2}, phi, {0,0}, subtract_from(phi3a) );
        Conv.convolve_Hessians( phi, {0,2}, phi, {0,2}, phi, {1,1}, subtract_from(phi3a) );
        Conv.convolve_Hessians( phi, {0,1}, phi, {0,1}, phi, {2,2}, subtract_from(phi3a) );
        phi3a.apply_InverseLaplacian();
        csoca::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime()-wtime << "s" << std::endl;
        
        //... 3b term ...
        wtime = get_wtime();    
        csoca::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(3b) term" << std::flush;
        phi3b.FourierTransformForward(false);
        Conv.convolve_SumOfHessians( phi, {0,0}, phi2, {1,1}, {2,2}, assign_to(phi3b) );
        Conv.convolve_SumOfHessians( phi, {1,1}, phi2, {2,2}, {0,0}, add_to(phi3b) );
        Conv.convolve_SumOfHessians( phi, {2,2}, phi2, {0,0}, {1,1}, add_to(phi3b) );
        Conv.convolve_Hessians( phi, {0,1}, phi2, {0,1}, subtract_twice_from(phi3b) );
        Conv.convolve_Hessians( phi, {0,2}, phi2, {0,2}, subtract_twice_from(phi3b) );
        Conv.convolve_Hessians( phi, {1,2}, phi2, {1,2}, subtract_twice_from(phi3b) );
        phi3b.apply_InverseLaplacian();
        phi3b *= 0.5; // factor 1/2 from definition of phi(3b)!
        csoca::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime()-wtime << "s" << std::endl;
        
        //... transversal term ...
        wtime = get_wtime();    
        csoca::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing A(3) term" << std::flush;
        for( int idim=0; idim<3; ++idim ){
            // cyclic rotations of indices
            int idimp = (idim+1)%3, idimpp = (idim+2)%3;
            A3[idim]->FourierTransformForward(false);
            Conv.convolve_Hessians( phi2, {idim,idimp},  phi, {idim,idimpp}, assign_to(*A3[idim]) );
            Conv.convolve_Hessians( phi2, {idim,idimpp}, phi, {idim,idimp},  subtract_from(*A3[idim]) );
            Conv.convolve_DifferenceOfHessians( phi, {idimp,idimpp}, phi2,{idimp,idimp}, {idimpp,idimpp}, add_to(*A3[idim]) );
            Conv.convolve_DifferenceOfHessians( phi2,{idimp,idimpp}, phi, {idimp,idimp}, {idimpp,idimpp}, subtract_from(*A3[idim]) );
            A3[idim]->apply_InverseLaplacian();
        }
        csoca::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime()-wtime << "s" << std::endl;
    }

    if( bSymplecticPT ){
        //... transversal term ...
        wtime = get_wtime();    
        csoca::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing vNLO(3) term" << std::flush;
        for( int idim=0; idim<3; ++idim ){
            // cyclic rotations of indices
            A3[idim]->FourierTransformForward(false);
            Conv.convolve_Gradient_and_Hessian( phi, {0},  phi2, {idim,0}, assign_to(*A3[idim]) );
            Conv.convolve_Gradient_and_Hessian( phi, {1},  phi2, {idim,1}, add_to(*A3[idim]) );
            Conv.convolve_Gradient_and_Hessian( phi, {2},  phi2, {idim,2}, add_to(*A3[idim]) );
        }
        csoca::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime()-wtime << "s" << std::endl;

    }

    ///... scale all potentials with respective growth factors
    phi *= g1;
    phi2 *= g2;
    phi3a *= g3a; 
    phi3b *= g3b;
    (*A3[0]) *= g3c;
    (*A3[1]) *= g3c;
    (*A3[2]) *= g3c;

    csoca::ilog << "-----------------------------------------------------------------------------" << std::endl;
    
    // gadget2_output_interface gof( the_config );
    
    ///////////////////////////////////////////////////////////////////////
    // we store the densities here if we compute them
    //======================================================================
    const bool compute_densities = false;
    if( compute_densities ){
        const std::string fname_hdf5 = the_config.GetValueSafe<std::string>("output", "fname_hdf5", "output.hdf5");
        const std::string fname_analysis = the_config.GetValueSafe<std::string>("output", "fbase_analysis", "output");

        Grid_FFT<real_t> delta({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
        Grid_FFT<real_t> delta2({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
        Grid_FFT<real_t> delta3a({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
        Grid_FFT<real_t> delta3b({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
        Grid_FFT<real_t> delta3({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
        delta.FourierTransformForward(false);
        delta2.FourierTransformForward(false);
        delta3a.FourierTransformForward(false);
        delta3b.FourierTransformForward(false);
        delta3.FourierTransformForward(false);

        #pragma omp parallel for
        for (size_t i = 0; i < phi.size(0); ++i)
        {
            for (size_t j = 0; j < phi.size(1); ++j)
            {
                for (size_t k = 0; k < phi.size(2); ++k)
                {
                    auto kk = phi.get_k<real_t>(i,j,k);
                    size_t idx = phi.get_idx(i,j,k);
                    auto laplace = -kk.norm_squared();

                    // compute densities associated to respective potentials as well
                    delta.kelem(idx) = laplace * phi.kelem(idx);
                    delta2.kelem(idx) = laplace * phi2.kelem(idx);
                    delta3a.kelem(idx) = laplace * phi3a.kelem(idx);
                    delta3b.kelem(idx) = laplace * phi3b.kelem(idx);
                    delta3.kelem(idx) = delta3a.kelem(idx) + delta3b.kelem(idx);
                }
            }
        }

        delta.Write_PowerSpectrum(fname_analysis+"_"+"power_delta1.txt");
        delta2.Write_PowerSpectrum(fname_analysis+"_"+"power_delta2.txt");
        delta3a.Write_PowerSpectrum(fname_analysis+"_"+"power_delta3a.txt");
        delta3b.Write_PowerSpectrum(fname_analysis+"_"+"power_delta3b.txt");
        delta3.Write_PowerSpectrum(fname_analysis+"_"+"power_delta3.txt");

        phi.FourierTransformBackward();
        phi2.FourierTransformBackward();
        phi3a.FourierTransformBackward();
        phi3b.FourierTransformBackward();

        delta.FourierTransformBackward();
        delta2.FourierTransformBackward();
        delta3a.FourierTransformBackward();
        delta3b.FourierTransformBackward();
        delta3.FourierTransformBackward();

        A3[0]->FourierTransformBackward();
        A3[1]->FourierTransformBackward();
        A3[2]->FourierTransformBackward();

#if defined(USE_MPI)
        if( CONFIG::MPI_task_rank == 0 )
            unlink(fname_hdf5.c_str());
        MPI_Barrier( MPI_COMM_WORLD );
#else
        unlink(fname_hdf5.c_str());
#endif

        phi.Write_to_HDF5(fname_hdf5, "phi");
        phi2.Write_to_HDF5(fname_hdf5, "phi2");
        phi3a.Write_to_HDF5(fname_hdf5, "phi3a");
        phi3b.Write_to_HDF5(fname_hdf5, "phi3b");
        
        delta.Write_to_HDF5(fname_hdf5, "delta");
        delta2.Write_to_HDF5(fname_hdf5, "delta2");
        delta3a.Write_to_HDF5(fname_hdf5, "delta3a");
        delta3b.Write_to_HDF5(fname_hdf5, "delta3b");
        delta3.Write_to_HDF5(fname_hdf5, "delta3");

        A3[0]->Write_to_HDF5(fname_hdf5, "A3x");
        A3[1]->Write_to_HDF5(fname_hdf5, "A3y");
        A3[2]->Write_to_HDF5(fname_hdf5, "A3z");

    }else{
        //======================================================================
        // we store displacements and velocities here if we compute them
        //======================================================================
        Grid_FFT<real_t> tmp({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
        size_t num_p_in_load = phi.local_size();
        particle_container particles;

        // if output plugin wants particles, then we need to store them, along with their IDs
        if( the_output_plugin->write_species_as_particles( cosmo_species::dm ) ){
            // if particles occupy a bcc lattice, then there are 2 x N^3 of them...
            if( initial_bcc_lattice )
                particles.allocate( 2*num_p_in_load );
            else
                particles.allocate( num_p_in_load );
            
            // generate particle IDs
            auto ipcount0 = (initial_bcc_lattice? 2:1) * particles.get_local_offset();
            for( size_t i=0,ipcount=0; i<tmp.size(0); ++i ){
                for( size_t j=0; j<tmp.size(1); ++j){
                    for( size_t k=0; k<tmp.size(2); ++k){
                        particles.set_id( ipcount, ipcount+ipcount0 );
                        ++ipcount;
                        if( initial_bcc_lattice ){
                            particles.set_id( ipcount, ipcount+ipcount0 );
                            ++ipcount;    
                        }
                    }
                }
            }
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
                        auto kk = phi.get_k<real_t>(i,j,k);
                        size_t idx = phi.get_idx(i,j,k);
                        auto phitot = phi.kelem(idx) + phi2.kelem(idx) + phi3a.kelem(idx) + phi3b.kelem(idx);
                        // divide by Lbox, because displacement is in box units for output plugin
                        tmp.kelem(idx) = lunit * ccomplex_t(0.0,1.0) * (kk[idim] * phitot + kk[idimp] * A3[idimpp]->kelem(idx) - kk[idimpp] * A3[idimp]->kelem(idx) ) / boxlen;
                    }
                }
            }
            tmp.FourierTransformBackward();

            // if we write particle data, store particle data in particle structure
            if( the_output_plugin->write_species_as_particles( cosmo_species::dm ) ){
                for( size_t i=0,ipcount=0; i<tmp.size(0); ++i ){
                    for( size_t j=0; j<tmp.size(1); ++j){
                        for( size_t k=0; k<tmp.size(2); ++k){
                            auto pos = tmp.get_unit_r<float>(i,j,k);
                            particles.set_pos( ipcount++, idim, pos[idim]*lunit + tmp.relem(i,j,k) );
                        }
                    }
                }

                if( initial_bcc_lattice ){
                    tmp.stagger_field();
                    auto ipcount0 = num_p_in_load;
                    for( size_t i=0,ipcount=ipcount0; i<tmp.size(0); ++i ){
                        for( size_t j=0; j<tmp.size(1); ++j){
                            for( size_t k=0; k<tmp.size(2); ++k){
                                auto pos = tmp.get_unit_r_staggered<float>(i,j,k);
                                particles.set_pos( ipcount++, idim, pos[idim]*lunit + tmp.relem(i,j,k) );
                            }
                        }
                    }
                }
            } 
            // otherwise write out the grid data directly to the output plugin
            else if( the_output_plugin->write_species_as_grid( cosmo_species::dm ))
            {
                fluid_component fc = (idim==0)? fluid_component::dx : ((idim==1)? fluid_component::dy : fluid_component::dz );
                the_output_plugin->write_grid_data( tmp, cosmo_species::dm, fc );
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
                        auto kk = phi.get_k<real_t>(i,j,k);
                        size_t idx = phi.get_idx(i,j,k);
                        // divide by Lbox, because displacement is in box units for output plugin
                        if(!bSymplecticPT){
                            auto phitot_v = vfac1 * phi.kelem(idx) + vfac2 * phi2.kelem(idx) + vfac3 * (phi3a.kelem(idx) + phi3b.kelem(idx));
                            tmp.kelem(idx) = vunit*ccomplex_t(0.0,1.0) * (kk[idim] * phitot_v + vfac3 * (kk[idimp] * A3[idimpp]->kelem(idx) - kk[idimpp] * A3[idimp]->kelem(idx)) ) / boxlen;
                        }else{
                            auto phitot_v = vfac1 * phi.kelem(idx) + vfac2 * phi2.kelem(idx);
                            tmp.kelem(idx) = vunit*ccomplex_t(0.0,1.0) * (kk[idim] * phitot_v) + vfac1 * A3[idim]->kelem(idx);
                        }
                    }
                }
            }
            tmp.FourierTransformBackward();

            // if we write particle data, store particle data in particle structure
            if( the_output_plugin->write_species_as_particles( cosmo_species::dm ) ){
                for( size_t i=0,ipcount=0; i<tmp.size(0); ++i ){
                    for( size_t j=0; j<tmp.size(1); ++j){
                        for( size_t k=0; k<tmp.size(2); ++k){
                            particles.set_vel( ipcount++, idim, tmp.relem(i,j,k) );
                        }
                    }
                }

                if( initial_bcc_lattice ){
                    tmp.stagger_field();
                    auto ipcount0 = num_p_in_load;
                    for( size_t i=0,ipcount=ipcount0; i<tmp.size(0); ++i ){
                        for( size_t j=0; j<tmp.size(1); ++j){
                            for( size_t k=0; k<tmp.size(2); ++k){
                                particles.set_vel( ipcount++, idim, tmp.relem(i,j,k) );
                            }
                        }
                    }
                }
            }// otherwise write out the grid data directly to the output plugin
            else if( the_output_plugin->write_species_as_grid( cosmo_species::dm ))
            {
                fluid_component fc = (idim==0)? fluid_component::vx : ((idim==1)? fluid_component::vy : fluid_component::vz );
                the_output_plugin->write_grid_data( tmp, cosmo_species::dm, fc );
            }
        }

        if( the_output_plugin->write_species_as_particles( cosmo_species::dm ) )
        {
            the_output_plugin->write_particle_data( particles, cosmo_species::dm );
        }
        
        if( the_output_plugin->write_species_as_grid( cosmo_species::baryon ) )
        {
            phi.FourierTransformForward();
            phi.apply_Laplacian();
            phi.FourierTransformBackward();
            the_output_plugin->write_grid_data( phi, cosmo_species::baryon, fluid_component::density );
        }
        

    }

    return 0;
}


} // end namespace ic_generator

