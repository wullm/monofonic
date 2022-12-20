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

#include <cosmology_parameters.hh>

/**
 * @brief namespace encapsulating all things cosmology
 * 
 */
namespace cosmology{

//! we store here the preset cosmological paramters
parameters::defaultmmap_t parameters::default_pmaps_
{
  //=============================================================================
  // Planck 2018 baseline cosmologies
  // cf. https://wiki.cosmos.esa.int/planck-legacy-archive/images/b/be/Baseline_params_table_2018_68pc.pdf
  //=============================================================================

  // baseline 2.17 base_plikHM_TTTEEE_lowl_lowE_lensing
  {"Planck2018EE", {
    {"h",           0.67321},
    {"Omega_m",     0.3158},
    {"Omega_b",     0.04938898},
    {"Omega_DE",    0.6842},
    {"Omega_k",     0.0},
    {"w_0",         -1.0},
    {"w_a",         0.0},
    {"n_s",         0.96605},
    {"A_s",         2.1005e-9},
    {"alpha_s",     0.0},
    {"beta_s",      0.0},
    {"k_p",         0.05},
    {"YHe",         0.245401},
    {"N_ur",        2.046},
    {"m_nu1",       0.06},
    {"m_nu2",       0.0},
    {"m_nu3",       0.0},
    {"deg_nu1",     1.0},
    {"deg_nu2",     1.0},
    {"deg_nu3",     1.0},
    {"Tcmb",        2.7255},
    {"O_nu_norm",   93.13861}}},

  // baseline 2.18 base_plikHM_TTTEEE_lowl_lowE_lensing_post_BAO
  {"Planck2018EE+BAO", {
    {"h",           0.67702},
    {"Omega_m",     0.3106},
    {"Omega_b",     0.04897284},
    {"Omega_DE",    0.6894},
    {"Omega_k",     0.0},
    {"w_0",         -1.0},
    {"w_a",         0.0},
    {"n_s",         0.96824},
    {"A_s",         2.1073e-9},
    {"alpha_s",     0.0},
    {"beta_s",      0.0},
    {"k_p",         0.05},
    {"YHe",         0.245425},
    {"N_ur",        2.046},
    {"m_nu1",       0.06},
    {"m_nu2",       0.0},
    {"m_nu3",       0.0},
    {"deg_nu1",     1.0},
    {"deg_nu2",     1.0},
    {"deg_nu3",     1.0},
    {"Tcmb",        2.7255},
    {"O_nu_norm",   93.13861}}},

  // baseline 2.19 base_plikHM_TTTEEE_lowl_lowE_lensing_post_Pantheon
  {"Planck2018EE+SN", {
    {"h",           0.6749},
    {"Omega_m",     0.3134},
    {"Omega_b",     0.04919537},
    {"Omega_DE",    0.6866},
    {"Omega_k",     0.0},
    {"w_0",         -1.0},
    {"w_a",         0.0},
    {"n_s",         0.96654},
    {"A_s",         2.1020e-9},
    {"alpha_s",     0.0},
    {"beta_s",      0.0},
    {"k_p",         0.05},
    {"YHe",         0.245411},
    {"N_ur",        2.046},
    {"m_nu1",       0.06},
    {"m_nu2",       0.0},
    {"m_nu3",       0.0},
    {"deg_nu1",     1.0},
    {"deg_nu2",     1.0},
    {"deg_nu3",     1.0},
    {"Tcmb",        2.7255},
    {"O_nu_norm",   93.13861}}},

  // baseline 2.20 base_plikHM_TTTEEE_lowl_lowE_lensing_post_BAO_Pantheon
  {"Planck2018EE+BAO+SN", {
    {"h",           0.67742},
    {"Omega_m",     0.3099},
    {"Omega_b",     0.048891054},
    {"Omega_DE",    0.6901},
    {"Omega_k",     0.0},
    {"w_0",         -1.0},
    {"w_a",         0.0},
    {"n_s",         0.96822},
    {"A_s",         2.1064e-9},
    {"alpha_s",     0.0},
    {"beta_s",      0.0},
    {"k_p",         0.05},
    {"YHe",         0.245421},
    {"N_ur",        2.046},
    {"m_nu1",       0.06},
    {"m_nu2",       0.0},
    {"m_nu3",       0.0},
    {"deg_nu1",     1.0},
    {"deg_nu2",     1.0},
    {"deg_nu3",     1.0},
    {"Tcmb",        2.7255},
    {"O_nu_norm",   93.13861}}}
};

/**
 * @brief Output all sets of cosmological parameters that we store internally
 * 
 */
void print_ParameterSets( void ){
  music::ilog << "Available cosmology parameter sets:" << std::endl;
  cosmology::parameters p;
  p.print_available_sets();
  music::ilog << std::endl;
}

//! Computes the relative factor for running of the spectral index
/*!
 * Note that the pivot scale and the argument k are in 1/Mpc.
 * Running does not affect the normalization at the pivot scale.
 * The power spectrum should be multiplied by the square of this factor.
 *
 * TODO: transfer functions and primordial spectra should be separated.
 */
real_t compute_running_factor(const cosmology::parameters *cosmo_params, real_t k)
{
    real_t alpha_s = cosmo_params->get("alpha_s");
    real_t beta_s = cosmo_params->get("beta_s");

    if (alpha_s == 0.0 && beta_s == 0.0) {
        return 1.0;
    } else {
        real_t k_p = cosmo_params->get("k_p"); // Mpc^-1
        real_t lnk = std::log(k / k_p);
        real_t running = 0.5 * (alpha_s * lnk + beta_s * lnk * lnk / 3.0);

        // one factor of (1/2) from the definition, another for the square root.
        return std::pow(k / k_p, 0.5 * running);
    }
}

}// end namespace cosmology