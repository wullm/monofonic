// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2020 by Oliver Hahn
// 
// monofonIC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// monofonIC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
