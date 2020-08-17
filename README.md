# MUSIC2 - monofonIC
Modular high-precision IC generator for cosmological simulations. MUSIC2-monofonIC is for non-zoom full box ICs (use [MUSIC](https://bitbucket.org/ohahn/music) for zooms, MUSIC2 for zooms is in the works).

[Full manual is available here as a wiki](https://bitbucket.org/ohahn/monofonic/wiki/). Quick instructions follow below.

Currently supported features (list is growing, so check back):

- Support for up to 3rd order Lagrangian perturbation theory (i.e. 1,2, and 3LPT)

- Multiple Einstein-Boltzmann modules: direct interface with [CLASS](https://lesgourg.github.io/class_public/class.html), file input from CAMB, and fitting formulae (Eisenstein&Hu).

- Multiple output modules for RAMSES, Arepo and Gadget-2/3 via plugins (Swift and Nyx are next). New codes can be added (see how to contribute in CONTRIBUTING.md file)

- Hybrid parallelization with MPI+OpenMP/threads.
    
- Requires FFTW v3, GSL (and HDF5 for output for some codes), as well as a CMake build system.

See file CONTRIBUTING.md on how to contribute to the development.


## Build Instructions
Clone code including submodules (currently only CLASS is used as a submodule):

    git clone --recurse-submodules https://<username>@bitbucket.org/ohahn/monofonic.git


Create build directory, configure, and build:

    mkdir monofonic/build; cd monofonic/build
	
    ccmake ..
	
    make

this should create an executable in the build directory. 

If you run into problems with CMake not being able to find your local FFTW3 or HDF5 installation, it is best to give the path directly as

    FFTW3_ROOT=<path> HDF5_ROOT=<path> ccmake ..

make sure to delete previous files generated by CMake before reconfiguring like this.

If you want to build on macOS, then it is strongly recommended to use GNU (or Intel) compilers instead of Apple's Clang. Install them e.g. 
via homebrew and then configure cmake to use them instead of the macOS default compiler via

    CC=gcc-9 CXX=g++-9 ccmake ..
    
This is necessary since Apple's compilers haven't supported OpenMP for years.

## Running

There is an example parameter file 'example.conf' in the main directory. Possible options are explained in it, it can be run
as a simple argument, e.g. from within the build directory:

     ./monofonic ../example.conf

If you want to run with MPI, you need to enable MPI support via ccmake. Then you can launch in hybrid MPI+threads mode by 
specifying the desired number of threads per task in the config file, and the number of tasks to be launched via

     mpirun -np 16 ./monofonic <path to config file>
     
It will then run with 16 tasks times the number of threads per task specified in the config file.