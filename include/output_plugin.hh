/*
 
 output.hh - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010  Oliver Hahn
 
*/

#pragma once

#include <string>
#include <cstring>
#include <map>
#include <numeric>

#include <particle_container.hh>
#include <general.hh>
#include <grid_fft.hh>
#include <config_file.hh>

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

	// virtual void prepare_output(const particle_container &pc) = 0;
	virtual void write_particle_data(const particle_container &pc) = 0;

	//! routine to return a multiplicative factor that contains the desired position units for the output
	virtual real_t position_unit() const = 0;

	//! routine to return a multiplicative factor that contains the desired velocity units for the output
	virtual real_t velocity_unit() const = 0;
	
};


#if 0
/*!
 * @class output_plugin
 * @brief abstract base class for output plug-ins
 *
 * This class provides the abstract base class for all output plug-ins.
 * All output plug-ins need to derive from it and implement the purely
 * virtual member functions.
 */
class output_plugin
{

public:
	using grid_hierarchy = Grid_FFT<real_t>;

protected:
	//! reference to the ConfigFile object that holds all configuration options
	ConfigFile &cf_;

	//! output file or directory name
	std::string fname_;

	//! minimum refinement level
	// unsigned levelmin_;

	//! maximum refinement level
	// unsigned levelmax_;

	std::vector<unsigned>
		offx_,  //!< vector describing the x-offset of each level
		offy_,  //!< vector describing the y-offset of each level
		offz_,  //!< vector describing the z-offset of each level
		sizex_, //!< vector describing the extent in x of each level
		sizey_, //!< vector describing the extent in y of each level
		sizez_; //!< vector describing the extent in z of each level

	//! quick access function to query properties of the refinement grid from the configuration options
	/*! @param name	name of the config property
	 *  @param icomp component index (0=x, 1=y, 2=z)
	 *  @param oit output iterator (e.g. std::back_inserter for vectors)
	 */
	template <typename output_iterator>
	void query_grid_prop(std::string name, int icomp, output_iterator oit)
	{
		char str[128];
		//for( unsigned i=levelmin_; i<=levelmax_; ++i )
		unsigned i = 0;
		{
			sprintf(str, "%s(%u,%d)", name.c_str(), i, icomp);
			*oit = 0; //cf_.GetValue<unsigned>( "setup", str );
			++oit;
		}
	}

public:
	//! constructor
	explicit output_plugin(ConfigFile &cf)
		: cf_(cf)
	{
		fname_ = cf_.GetValue<std::string>("output", "filename");
		// levelmin_	= cf_.GetValue<unsigned>( "setup", "levelmin" );
		// levelmax_	= cf_.GetValue<unsigned>( "setup", "levelmax" );

		query_grid_prop("offset", 0, std::back_inserter(offx_));
		query_grid_prop("offset", 1, std::back_inserter(offy_));
		query_grid_prop("offset", 2, std::back_inserter(offz_));

		query_grid_prop("size", 0, std::back_inserter(sizex_));
		query_grid_prop("size", 1, std::back_inserter(sizey_));
		query_grid_prop("size", 2, std::back_inserter(sizez_));
	}

	//! destructor
	virtual ~output_plugin()
	{
	}

	//! purely virtual prototype to write the masses for each dark matter particle
	virtual void write_dm_mass(const grid_hierarchy &gh) = 0;

	//! purely virtual prototype to write the dark matter density field
	virtual void write_dm_density(const grid_hierarchy &gh) = 0;

	//! purely virtual prototype to write the dark matter gravitational potential (from which displacements are computed in 1LPT)
	virtual void write_dm_potential(const grid_hierarchy &gh) = 0;

	//! purely virtual prototype to write dark matter particle velocities
	virtual void write_dm_velocity(int coord, const grid_hierarchy &gh) = 0;

	//! purely virtual prototype to write dark matter particle positions
	virtual void write_dm_position(int coord, const grid_hierarchy &gh) = 0;

	//! purely virtual prototype to write the baryon velocities
	virtual void write_gas_velocity(int coord, const grid_hierarchy &gh) = 0;

	//! purely virtual prototype to write the baryon coordinates
	virtual void write_gas_position(int coord, const grid_hierarchy &gh) = 0;

	//! purely virtual prototype to write the baryon density field
	virtual void write_gas_density(const grid_hierarchy &gh) = 0;

	//! purely virtual prototype to write the baryon gravitational potential (from which displacements are computed in 1LPT)
	virtual void write_gas_potential(const grid_hierarchy &gh) = 0;

	//! purely virtual prototype for all things to be done at the very end
	virtual void finalize(void) = 0;
};
#endif

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

