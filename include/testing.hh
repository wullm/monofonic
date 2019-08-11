#pragma once

#include <array>
#include <general.hh>
#include <config_file.hh>
#include <grid_fft.hh>

namespace testing{
    void output_potentials_and_densities( 
        ConfigFile& the_config,
        size_t ngrid, real_t boxlen,
        Grid_FFT<real_t>& phi,
        Grid_FFT<real_t>& phi2,
        Grid_FFT<real_t>& phi3a,
        Grid_FFT<real_t>& phi3b,
        std::array< Grid_FFT<real_t>*,3 >& A3 );
}
