#pragma once

#include <map>
#include <string>
#include <general.hh>
#include <config_file.hh>

enum tf_type
{
    total,
    cdm,
    baryon,
    vtotal,
    vcdm,
    vbaryon,
    total0
};

class TransferFunction_plugin
{
  public:
    // Cosmology cosmo_;    //!< cosmological parameter, read from config_file
    ConfigFile *pcf_;   //!< pointer to config_file from which to read parameters
    bool tf_distinct_;   //!< bool if density transfer function is distinct for baryons and DM
    bool tf_withvel_;    //!< bool if also have velocity transfer functions
    bool tf_withtotal0_; //!< have the z=0 spectrum for normalisation purposes
    bool tf_velunits_;   //!< velocities are in velocity units (km/s)
  public:
    //! constructor
    TransferFunction_plugin(ConfigFile &cf)
        : pcf_(&cf), tf_distinct_(false), tf_withvel_(false), tf_withtotal0_(false), tf_velunits_(false)
    {
        // real_t zstart;
        // zstart = pcf_->getValue<real_t>("setup", "zstart");
        // cosmo_.astart = 1.0 / (1.0 + zstart);
        // cosmo_.Omega_b = pcf_->getValue<real_t>("cosmology", "Omega_b");
        // cosmo_.Omega_m = pcf_->getValue<real_t>("cosmology", "Omega_m");
        // cosmo_.Omega_DE = pcf_->getValue<real_t>("cosmology", "Omega_L");
        // cosmo_.H0 = pcf_->getValue<real_t>("cosmology", "H0");
        // cosmo_.sigma8 = pcf_->getValue<real_t>("cosmology", "sigma_8");
        // cosmo_.nspect = pcf_->getValue<real_t>("cosmology", "nspec");
    }

    //! destructor
    virtual ~TransferFunction_plugin(){};

    //! compute value of transfer function at waven umber
    virtual double compute(double k, tf_type type) = 0;

    //! return maximum wave number allowed
    virtual double get_kmax(void) = 0;

    //! return minimum wave number allowed
    virtual double get_kmin(void) = 0;

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
    virtual TransferFunction_plugin *create(ConfigFile &cf) const = 0;

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
    TransferFunction_plugin *create(ConfigFile &cf) const
    {
        return new Derived(cf);
    }
};

// typedef TransferFunction_plugin TransferFunction;

TransferFunction_plugin *select_TransferFunction_plugin(ConfigFile &cf);
