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

#include <string>
#include <cstring>
#include <map>

#include <particle_container.hh>
#include <general.hh>
#include <grid_fft.hh>
#include <config_file.hh>
#include <cosmology_calculator.hh>

enum class output_type {particles,field_lagrangian,field_eulerian};


class output_plugin
{
protected:
	//! reference to the config_file object that holds all configuration options
	config_file &cf_;

	//! reference to the cosmology calculator object that does all things cosmological
	std::unique_ptr<cosmology::calculator> &pcc_;

	//! output file or directory name
	std::string fname_;

	//! name of the output interface
	std::string interface_name_;
public:
	//! constructor
	output_plugin(config_file &cf, std::unique_ptr<cosmology::calculator>& pcc, std::string interface_name )
		: cf_(cf), pcc_(pcc), interface_name_(interface_name)
	{
		fname_ = cf_.get_value<std::string>("output", "filename");
	}

	//! virtual destructor
	virtual ~output_plugin(){}

	//! routine to write particle data for a species
	virtual void write_particle_data(const particle::container &pc, const cosmo_species &s, double Omega_species ) {};

	//! routine to write gridded fluid component data for a species
	virtual void write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c ) {};

	//! routine to query whether species is written as grid data
	virtual output_type write_species_as ( const cosmo_species &s ) const = 0;

	//! routine to query whether species is written as grid data
	// virtual bool write_species_as_grid( const cosmo_species &s ) = 0;

	//! routine to query whether species is written as particle data
	// virtual bool write_species_as_particles( const cosmo_species &s ){ return !write_species_as_grid(s); }

	//! query if output wants 64bit precision for real values
	virtual bool has_64bit_reals() const = 0;

	//! query if output wants 64bit precision for integer values
	virtual bool has_64bit_ids() const = 0;
	
	//! routine to return a multiplicative factor that contains the desired position units for the output
	virtual real_t position_unit() const = 0;

	//! routine to return a multiplicative factor that contains the desired velocity units for the output
	virtual real_t velocity_unit() const = 0;

	//! routine to return a multiplicative factor that contains critical density * box volume in desired mass units for output
	virtual real_t mass_unit() const = 0;
};

/*!
 * @brief implements abstract factory design pattern for output plug-ins
 */
struct output_plugin_creator
{
	//! create an instance of a plug-in
	virtual std::unique_ptr<output_plugin> create(config_file &cf, std::unique_ptr<cosmology::calculator>& pcc) const = 0;

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
	std::unique_ptr<output_plugin> create(config_file &cf, std::unique_ptr<cosmology::calculator>& pcc) const
	{
		return std::make_unique<Derived>(cf,pcc); // Derived( cf );
	}
};

//! failsafe version to select the output plug-in
std::unique_ptr<output_plugin> select_output_plugin(config_file &cf, std::unique_ptr<cosmology::calculator>& pcc);

