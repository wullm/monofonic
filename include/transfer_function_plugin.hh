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
#include <memory>
#include <general.hh>
#include <config_file.hh>
#include <cosmology_parameters.hh>

enum tf_type
{
    delta_matter,
    delta_cdm,
    delta_baryon,
    theta_matter,
    theta_cdm,
    theta_baryon,
    delta_bc,
    theta_bc,
    delta_matter0,
    delta_cdm0,
    delta_baryon0,
    theta_matter0,
    theta_cdm0,
    theta_baryon0,
    delta_nu,
    delta_nu0,
    theta_nu,
    theta_nu0,
};

class TransferFunction_plugin
{
  public:
    config_file *pcf_;   //!< pointer to config_file from which to read parameters
    cosmology::parameters& cosmo_params_; //!< cosmological parameters are stored here
    bool tf_distinct_;   //!< bool if density transfer function is distinct for baryons and DM
    bool tf_withvel_;    //!< bool if also have velocity transfer functions
    bool tf_withtotal0_; //!< have the z=0 spectrum for normalisation purposes
    bool tf_velunits_;   //!< velocities are in velocity units (km/s)
    bool tf_isnormalised_; //!< assume that transfer functions come already correctly normalised and need be re-normalised to a specified value
    bool tf_with_asymptotic_growth_factors_; //!< has asymptotic growth factor/rate at large k
    
  public:
    //! constructor
    TransferFunction_plugin(config_file &cf, cosmology::parameters& cosmo_params)
        : pcf_(&cf), cosmo_params_( cosmo_params ), tf_distinct_(false), tf_withvel_(false), 
          tf_withtotal0_(false), tf_velunits_(false), tf_isnormalised_(false)
    { }

    //! destructor
    virtual ~TransferFunction_plugin(){};

    //! initialise, i.e. prepare data for later usage 
    virtual void intialise( void ) {}

    //! compute value of transfer function at waven umber
    virtual double compute(double k, tf_type type) const = 0;

    //! return maximum wave number allowed
    virtual double get_kmax(void) const = 0;

    //! return minimum wave number allowed
    virtual double get_kmin(void) const = 0;

    //! return asymptotic velocity factor vfac=aHf/h rate at large k
    virtual double get_vfac_asymptotic(void) const = 0;

    //! has asymptotic growth factor/rate
    bool tf_has_asymptotic_growth_factors(void)
    {
        return tf_with_asymptotic_growth_factors_;
    }

    //! return if density transfer function is distinct for baryons and DM
    bool tf_is_distinct(void)
    {
        return tf_distinct_;
    }

    //! return if we also have velocity transfer functions
    bool tf_has_velocities(void)
    {
        return tf_withvel_;
    }

    //! return if we also have a z=0 transfer function for normalisation
    bool tf_has_total0(void)
    {
        return tf_withtotal0_;
    }

    //! return if velocity returned is in velocity or in displacement units
    bool tf_velocity_units(void)
    {
        return tf_velunits_;
    }
};

//! Implements abstract factory design pattern for transfer function plug-ins
struct TransferFunction_plugin_creator
{
    //! create an instance of a transfer function plug-in
    virtual std::unique_ptr<TransferFunction_plugin> create(config_file &cf, cosmology::parameters& cp) const = 0;

    //! destroy an instance of a plug-in
    virtual ~TransferFunction_plugin_creator() {}
};

//! Write names of registered transfer function plug-ins to stdout
std::map<std::string, TransferFunction_plugin_creator *> &get_TransferFunction_plugin_map();
void print_TransferFunction_plugins(void);

//! Concrete factory pattern for transfer function plug-ins
template <class Derived>
struct TransferFunction_plugin_creator_concrete : public TransferFunction_plugin_creator
{
    //! register the plug-in by its name
    TransferFunction_plugin_creator_concrete(const std::string &plugin_name)
    {
        get_TransferFunction_plugin_map()[plugin_name] = this;
    }

    //! create an instance of the plug-in
    std::unique_ptr<TransferFunction_plugin> create(config_file &cf, cosmology::parameters& cp) const
    {
        return std::make_unique<Derived>(cf,cp);
    }
};

// typedef TransferFunction_plugin TransferFunction;

std::unique_ptr<TransferFunction_plugin> select_TransferFunction_plugin(config_file &cf, cosmology::parameters& cp);
