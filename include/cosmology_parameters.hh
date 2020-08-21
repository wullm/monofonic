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

#include <physical_constants.hh>
#include <config_file.hh>

namespace cosmology
{
//! structure for cosmological parameters
struct parameters
{

    double
        Omega_m,  //!< baryon+dark matter density
        Omega_b,  //!< baryon matter density
        Omega_DE, //!< dark energy density (cosmological constant or parameterised)
        Omega_r,  //!< photon + relativistic particle density
        Omega_k,  //!< curvature density
        f_b,      //!< baryon fraction
        H0,       //!< Hubble constant in km/s/Mpc
        h,        //!< hubble parameter
        nspect,   //!< long-wave spectral index (scale free is nspect=1)
        sigma8,   //!< power spectrum normalization
        Tcmb,     //!< CMB temperature (used to set Omega_r)
        Neff,     //!< effective number of neutrino species (used to set Omega_r)
        w_0,      //!< dark energy equation of state parameter 1: w = w0 + a * wa
        w_a,      //!< dark energy equation of state parameter 2: w = w0 + a * wa

        // below are helpers to store additional information
        dplus,     //!< linear perturbation growth factor
        f,         //!< growth factor logarithmic derivative
        pnorm,     //!< actual power spectrum normalisation factor
        sqrtpnorm, //!< sqrt of power spectrum normalisation factor
        vfact;     //!< velocity<->displacement conversion factor in Zel'dovich approx.

    parameters() = delete;
    
    parameters( const parameters& ) = default;
    
    explicit parameters(config_file cf)
    {
        H0 = cf.get_value<double>("cosmology", "H0");
        h  = H0 / 100.0;

        nspect = cf.get_value<double>("cosmology", "nspec");

        Omega_b = cf.get_value<double>("cosmology", "Omega_b");

        Omega_m = cf.get_value<double>("cosmology", "Omega_m");

        Omega_DE = cf.get_value<double>("cosmology", "Omega_L");

        w_0 = cf.get_value_safe<double>("cosmology", "w0", -1.0);

        w_a = cf.get_value_safe<double>("cosmology", "wa", 0.0);

        Tcmb = cf.get_value_safe<double>("cosmology", "Tcmb", 2.7255);

        Neff = cf.get_value_safe<double>("cosmology", "Neff", 3.046);

        sigma8 = cf.get_value_safe<double>("cosmology", "sigma_8",-1.0);

        // calculate energy density in ultrarelativistic species from Tcmb and Neff
        double Omega_gamma = 4 * phys_const::sigma_SI / std::pow(phys_const::c_SI, 3) * std::pow(Tcmb, 4.0) / phys_const::rhocrit_h2_SI / (h * h);
        double Omega_nu = Neff * Omega_gamma * 7. / 8. * std::pow(4. / 11., 4. / 3.);
        Omega_r = Omega_gamma + Omega_nu;

        if (cf.get_value_safe<bool>("cosmology", "ZeroRadiation", false))
        {
            Omega_r = 0.0;
        }

        f_b = Omega_b / Omega_m;
#if 1
        // assume zero curvature, take difference from dark energy
        Omega_DE += 1.0 - Omega_m - Omega_DE - Omega_r;
        Omega_k  = 0.0;
#else
        // allow for curvature 
        Omega_k = 1.0 - Omega_m - Omega_DE - Omega_r;
#endif

        dplus = 0.0;
        pnorm = 0.0;
        vfact = 0.0;

        music::ilog << "-------------------------------------------------------------------------------" << std::endl;
        music::ilog << "Cosmological parameters are: " << std::endl;
        music::ilog << " H0       = " << std::setw(16) << H0          << "sigma_8  = " << std::setw(16) << sigma8 << std::endl;
        music::ilog << " Omega_c  = " << std::setw(16) << Omega_m-Omega_b << "Omega_b  = " << std::setw(16) << Omega_b << std::endl;
        if (!cf.get_value_safe<bool>("cosmology", "ZeroRadiation", false)){
            music::ilog << " Omega_g  = " << std::setw(16) << Omega_gamma << "Omega_nu = " << std::setw(16) << Omega_nu << std::endl;
        }else{
            music::ilog << " Omega_r  = " << std::setw(16) << Omega_r << std::endl;
        }
        music::ilog << " Omega_DE = " << std::setw(16) << Omega_DE    << "nspect   = " << std::setw(16) << nspect << std::endl;
        music::ilog << " w0       = " << std::setw(16) << w_0         << "w_a      = " << std::setw(16) << w_a << std::endl;

        if( Omega_r > 0.0 )
        {
            music::wlog << "Radiation enabled, using Omega_r=" << Omega_r << " internally."<< std::endl;
            music::wlog << "Make sure your sim code supports this..." << std::endl;
        }
    }

};
} // namespace cosmology