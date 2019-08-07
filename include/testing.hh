#pragma once

namespace testing{
    void output_potentials_and_densities( 
        size_t ngrid, real_t boxlen,
        const Grid_FFT<real_t>& phi,
        const Grid_FFT<real_t>& phi2,
        const Grid_FFT<real_t>& phi3a,
        const Grid_FFT<real_t>& phi3b,
        const std::array< Grid_FFT<real_t>*,3 >& A3 );
}
