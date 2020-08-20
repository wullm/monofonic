/*******************************************************************\
 testing.hh - This file is part of MUSIC2 -
 a code to generate initial conditions for cosmological simulations 
 
 CHANGELOG (only majors, for details see repo):
    10/2019 - Michael Michaux & Oliver Hahn - first implementation
\*******************************************************************/
#pragma once

#include <array>
#include <general.hh>
#include <config_file.hh>
#include <grid_fft.hh>
#include <cosmology_calculator.hh>

namespace testing{
    void output_potentials_and_densities( 
        config_file& the_config,
        size_t ngrid, real_t boxlen,
        Grid_FFT<real_t>& phi,
        Grid_FFT<real_t>& phi2,
        Grid_FFT<real_t>& phi3,
        std::array< Grid_FFT<real_t>*,3 >& A3 );

    void output_velocity_displacement_symmetries(
        config_file &the_config,
        size_t ngrid, real_t boxlen, real_t vfac, real_t dplus,
        Grid_FFT<real_t> &phi,
        Grid_FFT<real_t> &phi2,
        Grid_FFT<real_t> &phi3,
        std::array<Grid_FFT<real_t> *, 3> &A3,
        bool bwrite_out_fields=false);

    void output_convergence(
        config_file &the_config,
        cosmology::calculator* the_cosmo_calc,
        std::size_t ngrid, real_t boxlen, real_t vfac, real_t dplus,
        Grid_FFT<real_t> &phi,
        Grid_FFT<real_t> &phi2,
        Grid_FFT<real_t> &phi3,
        std::array<Grid_FFT<real_t> *, 3> &A3);
}
