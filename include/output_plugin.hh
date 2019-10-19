/*
 
 output_plugin.hh - This file is part of MUSIC2 -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2019  Oliver Hahn
 
*/

#pragma once

#include <string>
#include <cstring>
#include <map>

#include <particle_container.hh>
#include <general.hh>
#include <grid_fft.hh>
#include <config_file.hh>

enum class output_type {particles,field_lagrangian,field_eulerian};

class output_plugin
{
protected:
	//! reference to the ConfigFile object that holds all configuration options
	ConfigFile &cf_;

	//! output file or directory name
	std::string fname_;

	//! name of the output interface
	std::string interface_name_;
public:
	//! constructor
	output_plugin(ConfigFile &cf, std::string interface_name )
		: cf_(cf), interface_name_(interface_name)
	{
		fname_ = cf_.GetValue<std::string>("output", "filename");
	}

	//! virtual destructor
	virtual ~output_plugin(){}

	//! routine to write particle data for a species
	virtual void write_particle_data(const particle::container &pc, const cosmo_species &s ) {};

	//! routine to write gridded fluid component data for a species
	virtual void write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c ) {};

	//! routine to query whether species is written as grid data
	virtual output_type write_species_as ( const cosmo_species &s ) const = 0;

	//! routine to query whether species is written as grid data
	// virtual bool write_species_as_grid( const cosmo_species &s ) = 0;

	//! routine to query whether species is written as particle data
	// virtual bool write_species_as_particles( const cosmo_species &s ){ return !write_species_as_grid(s); }
	
	//! routine to return a multiplicative factor that contains the desired position units for the output
	virtual real_t position_unit() const = 0;

	//! routine to return a multiplicative factor that contains the desired velocity units for the output
	virtual real_t velocity_unit() const = 0;
};

/*!
 * @brief implements abstract factory design pattern for output plug-ins
 */
struct output_plugin_creator
{
	//! create an instance of a plug-in
	virtual std::unique_ptr<output_plugin> create(ConfigFile &cf) const = 0;

	//! destroy an instance of a plug-in
	virtual ~output_plugin_creator() {}
};

//! maps the name of a plug-in to a pointer of the factory pattern
std::map<std::string, output_plugin_creator *> &get_output_plugin_map();

//! print a list of all registered output plug-ins
void print_output_plugins();

/*!
 * @brief concrete factory pattern for output plug-ins
 */
template <class Derived>
struct output_plugin_creator_concrete : public output_plugin_creator
{
	//! register the plug-in by its name
	output_plugin_creator_concrete(const std::string &plugin_name)
	{
		get_output_plugin_map()[plugin_name] = this;
	}

	//! create an instance of the plug-in
	std::unique_ptr<output_plugin> create(ConfigFile &cf) const
	{
		return std::make_unique<Derived>(cf); // Derived( cf );
	}
};

//! failsafe version to select the output plug-in
std::unique_ptr<output_plugin> select_output_plugin(ConfigFile &cf);

