#include <cmath>
#include <complex>
#include <iostream>
#include <fstream>
#include <thread>

#include <unistd.h> // for unlink

#include <general.hh>
#include <grid_fft.hh>

#include <transfer_function_plugin.hh>
#include <random_plugin.hh>
#include <output_plugin.hh>
#include <cosmology_calculator.hh>

namespace CONFIG{
int  MPI_thread_support = -1;
int  MPI_task_rank = 0;
int  MPI_task_size = 1;
bool MPI_ok = false;
bool MPI_threads_ok = false;
bool FFTW_threads_ok = false;
};

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
#endif

#if defined(USE_FFTW_THREADS)
  #if defined(USE_MPI)
    if (CONFIG::MPI_threads_ok)
        CONFIG::FFTW_threads_ok = fftw_init_threads();
  #else
    CONFIG::FFTW_threads_ok = fftw_init_threads();
  #endif 
#endif

#if defined(USE_FFTW_MPI)
    fftw_mpi_init();
    csoca::ilog << "MPI is enabled                : " << "yes" << std::endl;
#endif

    csoca::ilog << "MPI supports multi-threading  : " << CONFIG::MPI_threads_ok << std::endl;
    csoca::ilog << "FFTW supports multi-threading : " << CONFIG::FFTW_threads_ok << std::endl;
    csoca::ilog << "Available HW threads / task   : " << std::thread::hardware_concurrency() << std::endl;

    //------------------------------------------------------------------------------
    // Parse command line options
    //------------------------------------------------------------------------------

    if (argc != 2)
    {
        // print_region_generator_plugins();
        print_TransferFunction_plugins();
        // print_RNG_plugins();
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
    const real_t phifac = 1.0 / boxlen / boxlen; // to have potential in box units
    const real_t deriv_fac = 1.0 ;//boxlen;

    // real_t Dplus0 = the_config.GetValue<real_t>("setup", "Dplus0");
    // real_t Ddot0 = 1.0;

    const bool bDoFixing = false;

    
    
    //...
    const std::string fname_hdf5 = the_config.GetValueSafe<std::string>("output", "fname_hdf5", "output.hdf5");
    //////////////////////////////////////////////////////////////////////////////////////////////

    std::unique_ptr<CosmologyCalculator>
        the_cosmo_calc;
    

    try
    {
        the_random_number_generator = select_RNG_plugin(the_config);
        the_transfer_function       = select_TransferFunction_plugin(the_config);
        the_output_plugin           = select_output_plugin(the_config);

        the_cosmo_calc = std::make_unique<CosmologyCalculator>(the_config, the_transfer_function);

        // double pnorm = the_cosmo_calc->ComputePNorm();        
        // csoca::ilog << "power spectrum is output for D+ =" << Dplus0 << std::endl;
        //csoca::ilog << "power spectrum normalisation is " << pnorm << std::endl;
        //csoca::ilog << "power spectrum normalisation is " << pnorm*Dplus*Dplus << std::endl;

    }catch(...){
        csoca::elog << "Problem during initialisation. See error(s) above. Exiting..." << std::endl;
        #if defined(USE_MPI) 
        MPI_Finalize();
        #endif
        return 1;
    }
    const real_t Dplus0 = the_cosmo_calc->CalcGrowthFactor(astart) / the_cosmo_calc->CalcGrowthFactor(1.0);
    const real_t vfac   = the_cosmo_calc->CalcVFact(astart);

    {
        // write power spectrum to a file
        std::ofstream ofs("input_powerspec.txt");
        for( double k=1e-4; k<1e4; k*=1.1 ){
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
    
    phi.FillRandomReal(6519);

    //======================================================================
    //... compute 1LPT displacement potential ....
    // phi = - delta / k^2
    phi.FourierTransformForward();
    
    phi.apply_function_k_dep([&](auto x, auto k) -> ccomplex_t {
        real_t kmod = k.norm();
        if( bDoFixing ) x = x / std::abs(x); //std::exp(ccomplex_t(0, iphase * PhaseRotation));
        else x = x;
        ccomplex_t delta = x * the_cosmo_calc->GetAmplitude(kmod, total);
        return -delta / (kmod * kmod) * phifac / volfac;
    });

    phi.zero_DC_mode();

    //======================================================================
    //... compute 2LPT displacement potential
    Grid_FFT<real_t>
        phi_xx({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}),
        phi_xy({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}),
        phi_xz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}),
        phi_yy({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}),
        phi_yz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}),
        phi_zz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    phi_xx.FourierTransformForward(false);
    phi_xy.FourierTransformForward(false);
    phi_xz.FourierTransformForward(false);
    phi_yy.FourierTransformForward(false);
    phi_yz.FourierTransformForward(false);
    phi_zz.FourierTransformForward(false);

    #pragma omp parallel for
    for (size_t i = 0; i < phi.size(0); ++i)
    {
        for (size_t j = 0; j < phi.size(1); ++j)
        {
            for (size_t k = 0; k < phi.size(2); ++k)
            {
                auto kk = phi.get_k<real_t>(i,j,k);
                size_t idx = phi.get_idx(i,j,k);

                phi_xx.kelem(idx) = -kk[0] * kk[0] * phi.kelem(idx) / phifac;
                phi_xy.kelem(idx) = -kk[0] * kk[1] * phi.kelem(idx) / phifac;
                phi_xz.kelem(idx) = -kk[0] * kk[2] * phi.kelem(idx) / phifac;
                phi_yy.kelem(idx) = -kk[1] * kk[1] * phi.kelem(idx) / phifac;
                phi_yz.kelem(idx) = -kk[1] * kk[2] * phi.kelem(idx) / phifac;
                phi_zz.kelem(idx) = -kk[2] * kk[2] * phi.kelem(idx) / phifac;
            }
        }
    }

    phi_xx.FourierTransformBackward();
    phi_xy.FourierTransformBackward();
    phi_xz.FourierTransformBackward();
    phi_yy.FourierTransformBackward();
    phi_yz.FourierTransformBackward();
    phi_zz.FourierTransformBackward();

    
    for (size_t i = 0; i < phi2.size(0); ++i)
    {
        for (size_t j = 0; j < phi2.size(1); ++j)
        {
            for (size_t k = 0; k < phi2.size(2); ++k)
            {
                size_t idx = phi2.get_idx(i, j, k);

                phi2.relem(idx) =  phi_xx.relem(idx)*phi_yy.relem(idx)-phi_xy.relem(idx)*phi_xy.relem(idx)
                                  +phi_xx.relem(idx)*phi_zz.relem(idx)-phi_xz.relem(idx)*phi_xz.relem(idx)
                                  +phi_yy.relem(idx)*phi_zz.relem(idx)-phi_yz.relem(idx)*phi_yz.relem(idx);
            }
        }
    }

    phi2.FourierTransformForward();
    phi2.apply_function_k_dep([&](auto x, auto k) {
        real_t kmod2 = k.norm_squared();
        return x * (-1.0 / kmod2) * phifac;
    });
    phi2.zero_DC_mode();
    
    //======================================================================
    //... compute 3LPT displacement potential
    Grid_FFT<real_t>
        phi2_xx({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}),
        phi2_xy({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}),
        phi2_xz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}),
        phi2_yy({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}),
        phi2_yz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}),
        phi2_zz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    phi2_xx.FourierTransformForward(false);
    phi2_xy.FourierTransformForward(false);
    phi2_xz.FourierTransformForward(false);
    phi2_yy.FourierTransformForward(false);
    phi2_yz.FourierTransformForward(false);
    phi2_zz.FourierTransformForward(false);

    #pragma omp parallel for
    for (size_t i = 0; i < phi2.size(0); ++i)
    {
        for (size_t j = 0; j < phi2.size(1); ++j)
        {
            for (size_t k = 0; k < phi2.size(2); ++k)
            {
                auto kk = phi2.get_k<real_t>(i,j,k);
                size_t idx = phi2.get_idx(i,j,k);

                phi2_xx.kelem(idx) = -kk[0] * kk[0] * phi2.kelem(idx) / phifac;
                phi2_xy.kelem(idx) = -kk[0] * kk[1] * phi2.kelem(idx) / phifac;
                phi2_xz.kelem(idx) = -kk[0] * kk[2] * phi2.kelem(idx) / phifac;
                phi2_yy.kelem(idx) = -kk[1] * kk[1] * phi2.kelem(idx) / phifac;
                phi2_yz.kelem(idx) = -kk[1] * kk[2] * phi2.kelem(idx) / phifac;
                phi2_zz.kelem(idx) = -kk[2] * kk[2] * phi2.kelem(idx) / phifac;
            }
        }
    }

    phi2_xx.FourierTransformBackward();
    phi2_xy.FourierTransformBackward();
    phi2_xz.FourierTransformBackward();
    phi2_yy.FourierTransformBackward();
    phi2_yz.FourierTransformBackward();
    phi2_zz.FourierTransformBackward();

    for (size_t i = 0; i < phi3a.size(0); ++i)
    {
        for (size_t j = 0; j < phi3a.size(1); ++j)
        {
            for (size_t k = 0; k < phi3a.size(2); ++k)
            {
                size_t idx = phi3a.get_idx(i, j, k);

                phi3a.relem(idx) = 0.5 * (
                    + phi_xx.relem(idx) * ( phi2_yy.relem(idx) + phi2_zz.relem(idx) )
                    + phi_yy.relem(idx) * ( phi2_zz.relem(idx) + phi2_xx.relem(idx) )
                    + phi_zz.relem(idx) * ( phi2_xx.relem(idx) + phi2_yy.relem(idx) )
                    - phi_xy.relem(idx) * phi2_xy.relem(idx) * 2.0
                    - phi_xz.relem(idx) * phi2_xz.relem(idx) * 2.0
                    - phi_yz.relem(idx) * phi2_yz.relem(idx) * 2.0
                );
                
                phi3b.relem(idx) = 
                    + phi_xx.relem(idx)*phi_yy.relem(idx)*phi_zz.relem(idx)
                    + phi_xy.relem(idx)*phi_xz.relem(idx)*phi_yz.relem(idx) * 2.0
                    - phi_yz.relem(idx)*phi_yz.relem(idx)*phi_xx.relem(idx)
                    - phi_xz.relem(idx)*phi_xz.relem(idx)*phi_yy.relem(idx)
                    - phi_xy.relem(idx)*phi_xy.relem(idx)*phi_zz.relem(idx);
            }
        }
    }

    phi3a.FourierTransformForward();
    phi3a.apply_function_k_dep([&](auto x, auto k) {
        real_t kmod2 = k.norm_squared();
        return x * (-1.0 / kmod2) * phifac;
    });
    phi3a.zero_DC_mode();
    
    phi3b.FourierTransformForward();
    phi3b.apply_function_k_dep([&](auto x, auto k) {
        real_t kmod2 = k.norm_squared();
        return x * (-1.0 / kmod2) * phifac;
    });
    phi3b.zero_DC_mode();
    
    ///////////////////////////////////////////////////////////////////////
    // we store the densities here if we compute them
    Grid_FFT<real_t> &delta = phi_xx;
    Grid_FFT<real_t> &delta2 = phi_xy;
    Grid_FFT<real_t> &delta3a = phi_xz;
    Grid_FFT<real_t> &delta3b = phi_yy;
    Grid_FFT<real_t> &delta3 = phi_yz;

    // we store displacements and velocities here if we compute them
    Grid_FFT<real_t> &Psix = phi_xx;
    Grid_FFT<real_t> &Psiy = phi_xy;
    Grid_FFT<real_t> &Psiz = phi_xz;
    Grid_FFT<real_t> &Vx   = phi_yy;
    Grid_FFT<real_t> &Vy   = phi_yz;
    Grid_FFT<real_t> &Vz   = phi_zz;

    const bool compute_densities = false;
    
    phi_xx.FourierTransformForward(false);
    phi_xy.FourierTransformForward(false);
    phi_xz.FourierTransformForward(false);
    phi_yy.FourierTransformForward(false);
    phi_yz.FourierTransformForward(false);
    phi_zz.FourierTransformForward(false);

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

                if( compute_densities ){
                    // compute densities associated to respective potentials as well
                    delta.kelem(idx) = laplace * phi.kelem(idx) / phifac;
                    delta2.kelem(idx) = laplace * phi2.kelem(idx) / phifac;
                    delta3a.kelem(idx) = laplace * phi3a.kelem(idx) / phifac;
                    delta3b.kelem(idx) = laplace * phi3b.kelem(idx) / phifac;
                    delta3.kelem(idx) = delta3a.kelem(idx) + delta3b.kelem(idx);
                }else{
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
    }

    ///////////////////////////////////////////////////////////////////////
    

    //... write output .....
    

    if( compute_densities ){
        phi.FourierTransformBackward();
        phi2.FourierTransformBackward();
        phi3a.FourierTransformBackward();
        phi3b.FourierTransformBackward();

        delta.FourierTransformBackward();
        delta2.FourierTransformBackward();
        delta3a.FourierTransformBackward();
        delta3b.FourierTransformBackward();
        delta3.FourierTransformBackward();

        unlink(fname_hdf5.c_str());
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
    
    

#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif

    return 0;
}
