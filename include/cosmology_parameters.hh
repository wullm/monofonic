#pragma once

#include <config_file.hh>

//! structure for cosmological parameters
struct CosmologyParameters
{
    double
        Omega_m,  //!< baryon+dark matter density
        Omega_b,  //!< baryon matter density
        Omega_DE, //!< dark energy density (cosmological constant or parameterised)
        Omega_r,  //!< photon + relativistic particle density
        Omega_k,  //!< curvature density
        H0,       //!< Hubble constant in km/s/Mpc
        nspect,   //!< long-wave spectral index (scale free is nspect=1)
        sigma8,   //!< power spectrum normalization
        w_0,      //!< dark energy equation of state parameter 1: w = w0 + a * wa
        w_a,      //!< dark energy equation of state parameter 2: w = w0 + a * wa

        // below are helpers to store additional information
        dplus, //!< linear perturbation growth factor
        pnorm, //!< actual power spectrum normalisation factor
        sqrtpnorm, //!< sqrt of power spectrum normalisation factor
        vfact; //!< velocity<->displacement conversion factor in Zel'dovich approx.

    CosmologyParameters(ConfigFile cf)
    {
        Omega_b = cf.GetValue<double>("cosmology", "Omega_b");
        Omega_m = cf.GetValue<double>("cosmology", "Omega_m");
        Omega_DE = cf.GetValue<double>("cosmology", "Omega_L");
        w_0 = cf.GetValueSafe<double>("cosmology", "w0", -1.0);
        w_a = cf.GetValueSafe<double>("cosmology", "wa", 0.0);

        Omega_r = cf.GetValueSafe<double>("cosmology", "Omega_r", 0.0); // no longer default to nonzero (8.3e-5)
        Omega_k = 1.0 - Omega_m - Omega_DE - Omega_r;

        H0 = cf.GetValue<double>("cosmology", "H0");
        sigma8 = cf.GetValue<double>("cosmology", "sigma_8");
        nspect = cf.GetValue<double>("cosmology", "nspec");

        dplus = 0.0;
        pnorm = 0.0;
        vfact = 0.0;
    }

    CosmologyParameters(void)
    {
    }
};