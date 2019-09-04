# monofonIC

High order LPT/QPT tool for single resolution simulations

## Build Instructions
Clone code including submodules (currently only CLASS is used as a submodule):

    git clone --recurse-submodules https://ohahn@bitbucket.org/ohahn/monofonic.git


Create build directory, configure, and build:

    mkdir monofonic/build; cd monofonic/build
	
    ccmake ..
	
    make

this should create an executable in the build directory. 
There is an example parameter file 'example.conf' in the main directory
