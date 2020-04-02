#include <cmath>
#include <complex>
#include <iostream>
#include <fstream>
#include <thread>
#include <cfenv>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <general.hh>
#include <ic_generator.hh>
#include <particle_plt.hh>


// initialise with "default" values
namespace CONFIG{
int  MPI_thread_support = -1;
int  MPI_task_rank = 0;
int  MPI_task_size = 1;
bool MPI_ok = false;
bool MPI_threads_ok = false;
bool FFTW_threads_ok = false;
int  num_threads = 1;
}


#include "system_stat.hh"

#include <exception>
#include <stdexcept>
 
void handle_eptr(std::exception_ptr eptr) // passing by value is ok
{
    try {
        if (eptr) {
            std::rethrow_exception(eptr);
        }
    } catch(const std::exception& e) {
        csoca::elog << "This happened: \"" << e.what() << "\"" << std::endl;
    }
}

int main( int argc, char** argv )
{
    csoca::Logger::SetLevel(csoca::LogLevel::Info);
    // csoca::Logger::SetLevel(csoca::LogLevel::Debug);

    //------------------------------------------------------------------------------
    // initialise MPI 
    //------------------------------------------------------------------------------
    
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

    // Ascii ART logo. generated via http://patorjk.com/software/taag/#p=display&f=Nancyj&t=monofonIC
    csoca::ilog << "\n"
                << " The unigrid version of MUSIC-2         .8888b                   dP  a88888b. \n"
                << "                                        88   \"                   88 d8\'   `88 \n"
                << "  88d8b.d8b. .d8888b. 88d888b. .d8888b. 88aaa  .d8888b. 88d888b. 88 88        \n"
                << "  88\'`88\'`88 88\'  `88 88\'  `88 88\'  `88 88     88\'  `88 88\'  `88 88 88        \n"
                << "  88  88  88 88.  .88 88    88 88.  .88 88     88.  .88 88    88 88 Y8.   .88 \n"
                << "  dP  dP  dP `88888P\' dP    dP `88888P\' dP     `88888P\' dP    dP dP  Y88888P\' \n" << std::endl;

    // Compilation CMake configuration, time etc info:
    csoca::ilog << "This " << CMAKE_BUILDTYPE_STR << " build was compiled at " << __TIME__ << " on " <<  __DATE__ << std::endl;            
    
    // git and versioning info:
    csoca::ilog << "Version: v0.1a, git rev.: " << GIT_REV << ", tag: " << GIT_TAG << ", branch: " << GIT_BRANCH << std::endl;
    
    csoca::ilog << "-------------------------------------------------------------------------------" << std::endl;
    csoca::ilog << "Compile time options : " << std::endl;
    csoca::ilog << "                       Precision : " << CMAKE_PRECISION_STR << std::endl;
    csoca::ilog << "                    Convolutions : " << CMAKE_CONVOLVER_STR << std::endl;
    csoca::ilog << "                             PLT : " << CMAKE_PLT_STR << std::endl;
    csoca::ilog << "-------------------------------------------------------------------------------" << std::endl;


    //------------------------------------------------------------------------------
    // Parse command line options
    //------------------------------------------------------------------------------

    if (argc != 2)
    {
        // print_region_generator_plugins();
        print_TransferFunction_plugins();
        print_RNG_plugins();
        print_output_plugins();

        csoca::elog << "In order to run, you need to specify a parameter file!\n" << std::endl;
        exit(0);
    }

    // open the configuration file 
    ConfigFile the_config(argv[1]);

    //------------------------------------------------------------------------------
    // Set up FFTW
    //------------------------------------------------------------------------------

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

    CONFIG::num_threads = the_config.GetValueSafe<unsigned>("execution", "NumThreads",std::thread::hardware_concurrency());
    
#if defined(USE_FFTW_THREADS)
    if (CONFIG::FFTW_threads_ok)
        FFTW_API(plan_with_nthreads)(CONFIG::num_threads);
#endif

    //------------------------------------------------------------------------------
    // Set up OpenMP
    //------------------------------------------------------------------------------

#if defined(_OPENMP)
    omp_set_num_threads(CONFIG::num_threads);
#endif

    // std::feclearexcept(FE_ALL_EXCEPT);

    //------------------------------------------------------------------------------
    // Write code configuration to screen
    //------------------------------------------------------------------------------
    // hardware related infos
    csoca::ilog << std::setw(32) << std::left << "CPU vendor string" << " : " << SystemStat::Cpu().get_CPUstring() << std::endl;
    
    // multi-threading related infos
    csoca::ilog << std::setw(32) << std::left << "Available HW threads / task" << " : " << std::thread::hardware_concurrency() << " (" << CONFIG::num_threads << " used)" << std::endl;

    // memory related infos
    SystemStat::Memory mem;

    unsigned availpmem = mem.get_AvailMem()/1024/1024;
    unsigned usedpmem = mem.get_UsedMem()/1024/1024;
    unsigned maxpmem = availpmem, minpmem = availpmem;
    unsigned maxupmem = usedpmem, minupmem = usedpmem;
    
#if defined(USE_MPI)
    unsigned temp = 0;
    MPI_Allreduce(&minpmem,&temp,1,MPI_UNSIGNED,MPI_MIN,MPI_COMM_WORLD);  minpmem = temp;
    MPI_Allreduce(&maxpmem,&temp,1,MPI_UNSIGNED,MPI_MAX,MPI_COMM_WORLD);  maxpmem = temp;
    MPI_Allreduce(&minupmem,&temp,1,MPI_UNSIGNED,MPI_MIN,MPI_COMM_WORLD); minupmem = temp;
    MPI_Allreduce(&maxupmem,&temp,1,MPI_UNSIGNED,MPI_MAX,MPI_COMM_WORLD); maxupmem = temp;
#endif
    csoca::ilog << std::setw(32) << std::left << "Total system memory (phys)" << " : " << mem.get_TotalMem()/1024/1024 << " Mb" << std::endl;
    csoca::ilog << std::setw(32) << std::left << "Used system memory (phys)" << " : " << "Max: " << maxupmem << " Mb, Min: " << minupmem << " Mb" << std::endl;
    csoca::ilog << std::setw(32) << std::left << "Available system memory (phys)" << " : " <<  "Max: " << maxpmem << " Mb, Min: " << minpmem << " Mb" << std::endl;
    
    // MPI related infos
#if defined(USE_MPI)
    csoca::ilog << std::setw(32) << std::left << "MPI is enabled" << " : " << "yes (" << CONFIG::MPI_task_size << " tasks)" << std::endl;
    csoca::dlog << std::setw(32) << std::left << "MPI version" << " : " << MPI::get_version() << std::endl;
#else
    csoca::ilog << std::setw(32) << std::left << "MPI is enabled" << " : " << "no" << std::endl;
#endif
    csoca::ilog << std::setw(32) << std::left << "MPI supports multi-threading" << " : " << (CONFIG::MPI_threads_ok? "yes" : "no") << std::endl;
    
    // Kernel related infos
    SystemStat::Kernel kern;
    auto kinfo = kern.get_kernel_info();
    csoca::ilog << std::setw(32) << std::left << "OS/Kernel version" << " : " << kinfo.kernel << " version " << kinfo.major << "." << kinfo.minor << " build " << kinfo.build_number << std::endl;

    // FFTW related infos
    csoca::ilog << std::setw(32) << std::left << "FFTW version" << " : " << fftw_version << std::endl;
    csoca::ilog << std::setw(32) << std::left << "FFTW supports multi-threading" << " : " << (CONFIG::FFTW_threads_ok? "yes" : "no") << std::endl;
    csoca::ilog << std::setw(32) << std::left << "FFTW mode" << " : ";
#if defined(FFTW_MODE_PATIENT)
	csoca::ilog << "FFTW_PATIENT" << std::endl;
#elif defined(FFTW_MODE_MEASURE)
    csoca::ilog << "FFTW_MEASURE" << std::endl;
#else
	csoca::ilog << "FFTW_ESTIMATE" << std::endl;
#endif
    //--------------------------------------------------------------------
    // Initialise plug-ins
    //--------------------------------------------------------------------
    try
    {
        ic_generator::Initialise( the_config );
    }catch(...){
        handle_eptr( std::current_exception() );
        csoca::elog << "Problem during initialisation. See error(s) above. Exiting..." << std::endl;
        #if defined(USE_MPI) 
        MPI_Finalize();
        #endif
        return 1;
    }

    ///////////////////////////////////////////////////////////////////////
    // do the job...
    ///////////////////////////////////////////////////////////////////////
    ic_generator::Run( the_config );

    // particle::test_plt();
    ///////////////////////////////////////////////////////////////////////

#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif

    csoca::ilog << "-------------------------------------------------------------------------------" << std::endl;
    csoca::ilog << "Done. Have a nice day!\n" << std::endl;

    return 0;
}
