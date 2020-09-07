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

#include <map>
#include <string>

#include <physical_constants.hh>
#include <config_file.hh>

namespace cosmology
{
    //! singleton structure for cosmological parameters
    class parameters
    {
        /*double
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
        YHe,      //!< Helium fraction
        Neff,     //!< effective number of neutrino species (used to set Omega_r)
        w_0,      //!< dark energy equation of state parameter 1: w = w0 + a * wa
        w_a,      //!< dark energy equation of state parameter 2: w = w0 + a * wa

        // below are helpers to store additional information
        dplus,     //!< linear perturbation growth factor
        f,         //!< growth factor logarithmic derivative
        pnorm,     //!< actual power spectrum normalisation factor
        sqrtpnorm, //!< sqrt of power spectrum normalisation factor
        vfact;     //!< velocity<->displacement conversion factor in Zel'dovich approx.
*/
    private:
        std::map<std::string, double> pmap_;

    public:
        double get(const std::string &key) const
        {
            auto it = pmap_.find(key);
            if (it == pmap_.end())
            {
                auto errmsg = std::string("Cosmological parameter \'") + key + std::string("\' does not exist in internal list.");
                music::elog << errmsg << std::endl;
                throw std::runtime_error(errmsg.c_str());
            }
            return it->second;
        }

        void set(const std::string &key, const double value)
        {
            auto it = pmap_.find(key);
            if (it != pmap_.end())
            {
                pmap_[key] = value;
            }
            else
            {
                auto errmsg = std::string("Cosmological parameter \'") + key + std::string("\' does not exist in internal list. Needs to be defaulted before it can be set!");
                music::elog << errmsg << std::endl;
                throw std::runtime_error(errmsg.c_str());
            }
        }

        inline double operator[](const std::string &key) const { return this->get(key); }

        parameters() = delete;

        parameters(const parameters &) = default;

        explicit parameters(config_file &cf)
        {
            // CMB
            pmap_["Tcmb"] = cf.get_value_safe<double>("cosmology", "Tcmb", 2.7255);
            pmap_["YHe"] = cf.get_value_safe<double>("cosmology", "YHe", 0.2454006);

            // H0
            pmap_["H0"] = cf.get_value<double>("cosmology", "H0");
            pmap_["h"] = cf.get_value<double>("cosmology", "H0") / 100.0;
            const double h = pmap_["h"];

            // primordial and normalisation
            if(!cf.contains_key("cosmology/n_s"))
                pmap_["n_s"] = cf.get_value<double>("cosmology", "nspec");
            else
                pmap_["n_s"] = cf.get_value<double>("cosmology", "n_s");
            pmap_["A_s"] = cf.get_value_safe<double>("cosmology", "A_s", -1.0);
            pmap_["k_p"] = cf.get_value_safe<double>("cosmology", "k_p", 0.05);
            pmap_["sigma_8"] = cf.get_value_safe<double>("cosmology", "sigma_8", -1.0);
            
            // baryon and non-relativistic matter content
            pmap_["Omega_b"] = cf.get_value<double>("cosmology", "Omega_b");
            pmap_["Omega_m"] = cf.get_value<double>("cosmology", "Omega_m");

            // massive neutrino species
            pmap_["m_nu1"] = cf.get_value_safe<double>("cosmology", "m_nu1", 0.06);
            pmap_["m_nu2"] = cf.get_value_safe<double>("cosmology", "m_nu2", 0.0);
            pmap_["m_nu3"] = cf.get_value_safe<double>("cosmology", "m_nu3", 0.0);
            pmap_["N_nu_massive"] = int(this->get("m_nu1") > 1e-9) + int(this->get("m_nu2") > 1e-9) + int(this->get("m_nu3") > 1e-9);
            const double sum_m_nu = this->get("m_nu1") + this->get("m_nu2") + this->get("m_nu3");

            // number ultrarelativistic neutrinos
            pmap_["N_ur"] = cf.get_value_safe<double>("cosmology", "N_ur", 3.046 - this->get("N_nu_massive"));
            pmap_["Omega_nu_massive"] = sum_m_nu / (93.14 * h * h); // Omega_nu_m = \sum_i m_i / (93.14 eV h^2)

            // compute amount of cold dark matter as the rest
            pmap_["Omega_c"] = this->get("Omega_m") - this->get("Omega_b") - this->get("Omega_nu_massive");

            // calculate energy density in ultrarelativistic species from Tcmb and Neff
            // photons
            pmap_["Omega_gamma"] = 4 * phys_const::sigma_SI / std::pow(phys_const::c_SI, 3) * std::pow(this->get("Tcmb"), 4.0) 
                    / phys_const::rhocrit_h2_SI / (this->get("h") * this->get("h"));
            // massless neutrinos
            pmap_["Omega_nu_massless"] = this->get("N_ur") * this->get("Omega_gamma") * 7. / 8. * std::pow(4. / 11., 4. / 3.);
            // total relativistic
            pmap_["Omega_r"] = this->get("Omega_gamma") + this->get("Omega_nu_massless");

            // dark energy
            pmap_["Omega_DE"] = cf.get_value<double>("cosmology", "Omega_L");
            pmap_["w_0"] = cf.get_value_safe<double>("cosmology", "w_0", -1.0);
            pmap_["w_a"] = cf.get_value_safe<double>("cosmology", "w_a", 0.0);

            if (cf.get_value_safe<bool>("cosmology", "ZeroRadiation", false))
            {
                pmap_["Omega_r"] = 0.0;
            }

            pmap_["f_b"] = this->get("Omega_b") / this->get("Omega_m");
            pmap_["f_c"] = 1.0 - this->get("f_b"); // this means we add massive neutrinos to CDM here

#if 1
            // assume zero curvature, take difference from dark energy
            pmap_["Omega_DE"] += 1.0 - this->get("Omega_m") - this->get("Omega_DE") - this->get("Omega_r");
            // Omega_DE += 1.0 - Omega_m - Omega_DE - Omega_r;
            pmap_["Omega_k"] = 0.0;
#else
            // allow for curvature
            Omega_k = 1.0 - Omega_m - Omega_DE - Omega_r;
#endif

            pmap_["dplus"] = 0.0;
            pmap_["pnorm"] = 0.0;
            pmap_["sqrtpnorm"] = 0.0;
            pmap_["vfact"] = 0.0;

            music::ilog << "-------------------------------------------------------------------------------" << std::endl;
            music::ilog << "Cosmological parameters are: " << std::endl;
            music::ilog << " h        = " << std::setw(16) << this->get("h");
            if( this->get("A_s") > 0.0 )
              music::ilog << "A_s      = " << std::setw(16) << this->get("A_s");
            else
              music::ilog << "sigma_8  = " << std::setw(16) << this->get("sigma_8");
            music::ilog << "nspec   = " << std::setw(16) << this->get("n_s") << std::endl;
            music::ilog << " Omega_c  = " << std::setw(16) << this->get("Omega_c")  << "Omega_b  = " << std::setw(16) << this->get("Omega_b") << "Omega_m = " << std::setw(16) << this->get("Omega_m") << std::endl;
            music::ilog << " Omega_r  = " << std::setw(16) << this->get("Omega_r")  << "Omega_nu = " << std::setw(16) << this->get("Omega_nu_massive") << "∑m_nu   = " << std::setw(11) << sum_m_nu << "eV" << std::endl;
            music::ilog << " Omega_DE = " << std::setw(16) << this->get("Omega_DE") << "w_0      = " << std::setw(16) << this->get("w_0")      << "w_a     = " << std::setw(16) << this->get("w_a") << std::endl;

            if (this->get("Omega_r") > 0.0)
            {
                music::wlog << " Radiation enabled, using Omega_r=" << this->get("Omega_r") << " internally for backscaling." << std::endl;
                music::wlog << " Make sure your sim code supports this, otherwise set [cosmology] / ZeroRadiation=true." << std::endl;
            }
        }
    };
} // namespace cosmology