#include <cmath>
#include <complex>
#include <iostream>
#include <fstream>
#include <thread>

#include <unistd.h> // for unlink

#include <general.hh>
#include <grid_fft.hh>
#include <convolution.hh>

#include <transfer_function_plugin.hh>
#include <random_plugin.hh>
#include <output_plugin.hh>
#include <cosmology_calculator.hh>

// initialise with "default" values
namespace CONFIG{
int  MPI_thread_support = -1;
int  MPI_task_rank = 0;
int  MPI_task_size = 1;
bool MPI_ok = false;
bool MPI_threads_ok = false;
bool FFTW_threads_ok = false;
}

RNG_plugin *the_random_number_generator;
TransferFunction_plugin *the_transfer_function;
output_plugin *the_output_plugin;

int main( int argc, char** argv )
{
    csoca::Logger::SetLevel(csoca::LogLevel::Info);
    // csoca::Logger::SetLevel(csoca::LogLevel::Debug);

    // initialise MPI and multi-threading
#if defined(USE_MPI)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &CONFIG::MPI_thread_support);
    CONFIG::MPI_threads_ok = CONFIG::MPI_thread_support >= MPI_THREAD_FUNNELED;
    MPI_Comm_rank(MPI_COMM_WORLD, &CONFIG::MPI_task_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &CONFIG::MPI_task_size);
    CONFIG::MPI_ok = true;

    // set up lower logging levels for other tasks
    if( CONFIG::MPI_task_rank!=0 )
    {
        csoca::Logger::SetLevel(csoca::LogLevel::Error);
    }
#endif

#if defined(USE_FFTW_THREADS)
  #if defined(USE_MPI)
    if (CONFIG::MPI_threads_ok)
        CONFIG::FFTW_threads_ok = FFTW_API(init_threads)();
  #else
    CONFIG::FFTW_threads_ok = FFTW_API(init_threads)();
  #endif 
#endif

#if defined(USE_MPI)
    FFTW_API(mpi_init)();
#endif
    
#if defined(USE_FFTW_THREADS)
    if (CONFIG::FFTW_threads_ok)
        FFTW_API(plan_with_nthreads)(std::thread::hardware_concurrency());
#endif

#if defined(USE_MPI)
    csoca::ilog << "MPI is enabled                : " << "yes (" << CONFIG::MPI_task_size << " tasks)" << std::endl;
#else
    csoca::ilog << "MPI is enabled                : " << "no" << std::endl;
#endif
    csoca::ilog << "MPI supports multi-threading  : " << (CONFIG::MPI_threads_ok? "yes" : "no") << std::endl;
    csoca::ilog << "Available HW threads / task   : " << std::thread::hardware_concurrency() << std::endl;
    csoca::ilog << "FFTW supports multi-threading : " << (CONFIG::FFTW_threads_ok? "yes" : "no") << std::endl;
#if defined(FFTW_MODE_PATIENT)
	csoca::ilog << "FFTW mode                     : FFTW_PATIENT" << std::endl;
#elif defined(FFTW_MODE_MEASURE)
    csoca::ilog << "FFTW mode                     : FFTW_MEASURE" << std::endl;
#else
	csoca::ilog << "FFTW mode                     : FFTW_ESTIMATE" << std::endl;
#endif
    //------------------------------------------------------------------------------
    // Parse command line options
    //------------------------------------------------------------------------------

    if (argc != 2)
    {
        // print_region_generator_plugins();
        print_TransferFunction_plugins();
        print_RNG_plugins();
        print_output_plugins();

        csoca::elog << "In order to run, you need to specify a parameter file!" << std::endl;
        exit(0);
    }

    
    //--------------------------------------------------------------------
    // Initialise parameters
    ConfigFile the_config(argv[1]);

    const size_t ngrid = the_config.GetValue<size_t>("setup", "GridRes");
    const real_t boxlen = the_config.GetValue<double>("setup", "BoxLength");
    const real_t zstart = the_config.GetValue<double>("setup", "zstart");
    const int LPTorder = the_config.GetValueSafe<double>("setup","LPTorder",100);
    const real_t astart = 1.0/(1.0+zstart);
    const real_t volfac(std::pow(boxlen / ngrid / 2.0 / M_PI, 1.5));

    const bool bDoFixing = false;

    //...
    const std::string fname_hdf5 = the_config.GetValueSafe<std::string>("output", "fname_hdf5", "output.hdf5");
    const std::string fname_analysis = the_config.GetValueSafe<std::string>("output", "fbase_analysis", "output");
    //////////////////////////////////////////////////////////////////////////////////////////////

    std::unique_ptr<CosmologyCalculator>  the_cosmo_calc;
    
    try
    {
        the_random_number_generator = select_RNG_plugin(the_config);
        the_transfer_function       = select_TransferFunction_plugin(the_config);
        the_output_plugin           = select_output_plugin(the_config);
        the_cosmo_calc = std::make_unique<CosmologyCalculator>(the_config, the_transfer_function);
    }catch(...){
        csoca::elog << "Problem during initialisation. See error(s) above. Exiting..." << std::endl;
        #if defined(USE_MPI) 
        MPI_Finalize();
        #endif
        return 1;
    }
    const real_t Dplus0 = the_cosmo_calc->CalcGrowthFactor(astart) / the_cosmo_calc->CalcGrowthFactor(1.0);
    const real_t vfac   = the_cosmo_calc->CalcVFact(astart);

    if( CONFIG::MPI_task_rank==0 )
    {
        // write power spectrum to a file
        std::ofstream ofs("input_powerspec.txt");
        for( double k=the_transfer_function->get_kmin(); k<the_transfer_function->get_kmax(); k*=1.1 ){
            ofs << std::setw(16) << k
                << std::setw(16) << std::pow(the_cosmo_calc->GetAmplitude(k, total) * Dplus0, 2.0)
                << std::setw(16) << std::pow(the_cosmo_calc->GetAmplitude(k, total), 2.0)
                << std::endl;
        }
    }

    // compute growth factors of the respective orders
    const double g1  = -Dplus0;
    const double g2  = -3.0/7.0*Dplus0*Dplus0;
    const double g3a = -1.0/3.0*Dplus0*Dplus0*Dplus0;
    const double g3b = 10.0/21.*Dplus0*Dplus0*Dplus0;

    const double vfac1 =  vfac;
    const double vfac2 =  2*vfac1;
    const double vfac3 =  3*vfac1;

    //--------------------------------------------------------------------
    // Create arrays
    Grid_FFT<real_t> phi({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> phi2({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> phi3a({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> phi3b({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    OrszagConvolver<real_t> Conv({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    // NaiveConvolver<real_t> Conv({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    //--------------------------------------------------------------------
    // Some operators to add or subtract terms 
    auto assign_op = []( ccomplex_t res, ccomplex_t val ) -> ccomplex_t{ return res; };
    auto add_op = []( ccomplex_t res, ccomplex_t val ) -> ccomplex_t{ return val+res; };
    auto add2_op = []( ccomplex_t res, ccomplex_t val ) -> ccomplex_t{ return val+2.0*res; };
    auto sub_op = []( ccomplex_t res, ccomplex_t val ) -> ccomplex_t{ return val-res; };
    auto sub2_op = []( ccomplex_t res, ccomplex_t val ) -> ccomplex_t{ return val-2.0*res; };
    //--------------------------------------------------------------------
    
    //phi.FillRandomReal(6519);
    the_random_number_generator->Fill_Grid( phi );

    //======================================================================
    //... compute 1LPT displacement potential ....
    // phi = - delta / k^2
    double wtime = get_wtime();    
    csoca::ilog << "Computing phi(1) term..." << std::flush;
    phi.FourierTransformForward();
    
    phi.apply_function_k_dep([&](auto x, auto k) -> ccomplex_t {
        real_t kmod = k.norm();
        if( bDoFixing ) x = x / std::abs(x); //std::exp(ccomplex_t(0, iphase * PhaseRotation));
        else x = x;
        ccomplex_t delta = x * the_cosmo_calc->GetAmplitude(kmod, total);
        return -delta / (kmod * kmod) / volfac;
    });
    
    phi.zero_DC_mode();
    csoca::ilog << "   took " << get_wtime()-wtime << "s" << std::endl;
    
    //======================================================================
    //... compute 2LPT displacement potential ....
    
    wtime = get_wtime();    
    csoca::ilog << "Computing phi(2) term..." << std::flush;
    Conv.convolve_SumHessians( phi, {0,0}, phi, {1,1}, {2,2}, phi2, assign_op );
    Conv.convolve_Hessians( phi, {1,1}, phi, {2,2}, phi2, add_op );
    Conv.convolve_Hessians( phi, {0,1}, phi, {0,1}, phi2, sub_op );
    Conv.convolve_Hessians( phi, {0,2}, phi, {0,2}, phi2, sub_op );
    Conv.convolve_Hessians( phi, {1,2}, phi, {1,2}, phi2, sub_op );
    phi2.apply_InverseLaplacian();
    phi2.zero_DC_mode();
    csoca::ilog << "   took " << get_wtime()-wtime << "s" << std::endl;

    //======================================================================
    //... compute 3LPT displacement potential
    
    wtime = get_wtime();    
    csoca::ilog << "Computing phi(3a) term..." << std::flush;
    Conv.convolve_Hessians( phi, {0,0}, phi, {1,1}, phi, {2,2}, phi3a, assign_op );
    Conv.convolve_Hessians( phi, {0,1}, phi, {0,2}, phi, {1,2}, phi3a, add2_op );
    Conv.convolve_Hessians( phi, {1,2}, phi, {1,2}, phi, {0,0}, phi3a, sub_op );
    Conv.convolve_Hessians( phi, {0,2}, phi, {0,2}, phi, {1,1}, phi3a, sub_op );
    Conv.convolve_Hessians( phi, {0,1}, phi, {0,1}, phi, {2,2}, phi3a, sub_op );
    phi3a.apply_InverseLaplacian();
    csoca::ilog << "   took " << get_wtime()-wtime << "s" << std::endl;
    
    //...

    wtime = get_wtime();    
    csoca::ilog << "Computing phi(3b) term..." << std::flush;
    Conv.convolve_SumHessians( phi, {0,0}, phi2, {1,1}, {2,2}, phi3b, assign_op );
    Conv.convolve_SumHessians( phi, {1,1}, phi2, {2,2}, {0,0}, phi3b, add_op );
    Conv.convolve_SumHessians( phi, {2,2}, phi2, {0,0}, {1,1}, phi3b, add_op );
    Conv.convolve_Hessians( phi, {0,1}, phi2, {0,1}, phi3b, sub2_op );
    Conv.convolve_Hessians( phi, {0,2}, phi2, {0,2}, phi3b, sub2_op );
    Conv.convolve_Hessians( phi, {1,2}, phi2, {1,2}, phi3b, sub2_op );
    phi3b.apply_InverseLaplacian();
    phi3b *= 0.5; // factor 1/2 from definition of phi(3b)!
    csoca::ilog << "   took " << get_wtime()-wtime << "s" << std::endl;
    

    ///////////////////////////////////////////////////////////////////////
    // we store the densities here if we compute them
    const bool compute_densities = true;
    if( compute_densities ){
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

                    // scale potentials with respective order growth factors
                    phi.kelem(idx)   *= g1;
                    phi2.kelem(idx)  *= g2;
                    phi3a.kelem(idx) *= g3a;
                    phi3b.kelem(idx) *= g3b;

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
    }else{
        // we store displacements and velocities here if we compute them
        Grid_FFT<real_t> Psix({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
        Grid_FFT<real_t> Psiy({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
        Grid_FFT<real_t> Psiz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
        Grid_FFT<real_t> Vx({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
        Grid_FFT<real_t> Vy({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
        Grid_FFT<real_t> Vz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
        Psix.FourierTransformForward(false);
        Psiy.FourierTransformForward(false);
        Psiz.FourierTransformForward(false);
        Vx.FourierTransformForward(false);
        Vy.FourierTransformForward(false);
        Vz.FourierTransformForward(false);

        #pragma omp parallel for
        for (size_t i = 0; i < phi.size(0); ++i)
        {
            for (size_t j = 0; j < phi.size(1); ++j)
            {
                for (size_t k = 0; k < phi.size(2); ++k)
                {
                    auto kk = phi.get_k<real_t>(i,j,k);
                    size_t idx = phi.get_idx(i,j,k);

                    // scale potentials with respective order growth factors
                    phi.kelem(idx)   *= g1;
                    phi2.kelem(idx)  *= g2;
                    phi3a.kelem(idx) *= g3a;
                    phi3b.kelem(idx) *= g3b;
                    
                    auto phitot = phi.kelem(idx) + ((LPTorder>1)?phi2.kelem(idx):0.0) + ((LPTorder>2)? phi3a.kelem(idx) + phi3b.kelem(idx) : 0.0);
                    auto phitot_v = vfac1 * phi.kelem(idx) + ((LPTorder>1)? vfac2 * phi2.kelem(idx) : 0.0) + ((LPTorder>2)? vfac3 * (phi3a.kelem(idx) + phi3b.kelem(idx)) : 0.0);

                    Psix.kelem(idx) = ccomplex_t(0.0,1.0) * kk[0]* boxlen * ( phitot );
                    Psiy.kelem(idx) = ccomplex_t(0.0,1.0) * kk[1]* boxlen * ( phitot );
                    Psiz.kelem(idx) = ccomplex_t(0.0,1.0) * kk[2]* boxlen * ( phitot );

                    Vx.kelem(idx)   = ccomplex_t(0.0,1.0) * kk[0]* boxlen * ( phitot_v );
                    Vy.kelem(idx)   = ccomplex_t(0.0,1.0) * kk[1]* boxlen * ( phitot_v );
                    Vz.kelem(idx)   = ccomplex_t(0.0,1.0) * kk[2]* boxlen * ( phitot_v );
                }
            }
        }

        Psix.FourierTransformBackward();
        Psiy.FourierTransformBackward();
        Psiz.FourierTransformBackward();
        Vx.FourierTransformBackward();
        Vy.FourierTransformBackward();
        Vz.FourierTransformBackward();

        // Psix.Write_to_HDF5(fname_hdf5, "Psix");
        // Psiy.Write_to_HDF5(fname_hdf5, "Psiy");
        // Psiz.Write_to_HDF5(fname_hdf5, "Psiz");
        // Vx.Write_to_HDF5(fname_hdf5, "Vx");
        // Vy.Write_to_HDF5(fname_hdf5, "Vy");
        // Vz.Write_to_HDF5(fname_hdf5, "Vz");

        the_output_plugin->write_dm_mass(Psix);
        the_output_plugin->write_dm_density(Psix);

        the_output_plugin->write_dm_position(0, Psix );
        the_output_plugin->write_dm_position(1, Psiy );
        the_output_plugin->write_dm_position(2, Psiz );
        
        the_output_plugin->write_dm_velocity(0, Vx );
        the_output_plugin->write_dm_velocity(1, Vy );
        the_output_plugin->write_dm_velocity(2, Vz );
        
        the_output_plugin->finalize();
		delete the_output_plugin;
    }


    ///////////////////////////////////////////////////////////////////////
    

#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif

    return 0;
}
