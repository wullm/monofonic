/*
 
 output.hh - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010  Oliver Hahn
 
*/

#ifndef __OUTPUT_HH
#define __OUTPUT_HH

#include <string>
#include <cstring>
#include <map>
#include <numeric>

#include "general.hh"
#include "grid_fft.hh"
#include "config_file.hh"

constexpr int empty_fill_bytes{56};


class particle_container
{
public:
	std::vector<float> positions_, velocities_;
	std::vector<int> ids_;

	particle_container()
	{
	}

	particle_container(const particle_container &) = delete;

	const void* get_pos_ptr() const{
		return reinterpret_cast<const void*>( &positions_[0] );
	}

	const void* get_vel_ptr() const{
		return reinterpret_cast<const void*>( &velocities_[0] );
	}

	const void* get_ids_ptr() const{
		return reinterpret_cast<const void*>( &ids_[0] );
	}

	void allocate(size_t nump)
	{
		positions_.resize(3 * nump);
		velocities_.resize(3 * nump);
		ids_.resize(nump);
	}

	void set_pos(size_t ipart, size_t idim, real_t p)
	{
		positions_[3 * ipart + idim] = p;
	}

	void set_vel(size_t ipart, size_t idim, real_t p)
	{
		velocities_[3 * ipart + idim] = p;
	}

	void set_id(size_t ipart, id_t id)
	{
		ids_[ipart] = id;
	}

	size_t get_local_num_particles(void) const
	{
		return ids_.size();
	}

	size_t get_global_num_particles(void) const
	{
		size_t local_nump = ids_.size(), global_nump;
#ifdef USE_MPI
		MPI_Allreduce(reinterpret_cast<void *>(&local_nump), reinterpret_cast<void *>(&global_nump), 1,
					  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#else
		global_nump = local_nump;
#endif
		return global_nump;
	}

	size_t get_local_offset( void ) const
	{
		size_t this_offset = 0;

		#ifdef USE_MPI
			int mpi_size, mpi_rank;
			MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
			MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
			
			std::vector<size_t> nump_p_task(mpi_size,0), off_p_task;
			size_t num_p_this_task = this->get_local_num_particles();
			// MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
					//   void *recvbuf, int recvcount, MPI_Datatype recvtype,
					//   MPI_Comm comm)
			MPI_Allgather( reinterpret_cast<const void*>(&num_p_this_task), 1, MPI_UNSIGNED_LONG_LONG,
				reinterpret_cast<void*>(&nump_p_task[0]), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD );

			off_p_task.push_back( 0 );
			std::partial_sum(nump_p_task.begin(), nump_p_task.end(), std::back_inserter(off_p_task) );
			this_offset = nump_p_task.at(mpi_rank);
		#endif

		return this_offset;
	}

	void dump(void)
	{
		for (size_t i = 0; i < ids_.size(); ++i)
		{
			std::cout << positions_[3 * i + 0] << " " << positions_[3 * i + 1] << " " << positions_[3 * i + 2] << " "
					  << velocities_[3 * i + 0] << " " << velocities_[3 * i + 1] << " " << velocities_[3 * i + 2] << std::endl;
		}
	}
};

class output_interface
{
protected:
	//! reference to the ConfigFile object that holds all configuration options
	ConfigFile &cf_;

	//! output file or directory name
	std::string fname_;
	std::string interface_name_;
public:
	//! constructor
	output_interface(ConfigFile &cf, std::string interface_name )
		: cf_(cf), interface_name_(interface_name)
	{
		fname_ = cf_.GetValue<std::string>("output", "filename");
	}

	// virtual void prepare_output(const particle_container &pc) = 0;
	virtual void write_particle_data(const particle_container &pc) = 0;

	virtual real_t position_unit() const = 0;

	virtual real_t velocity_unit() const = 0;
	
};

class gadget2_output_interface : public output_interface
{
public:
	struct header
	{
		int npart[6];
		double mass[6];
		double time;
		double redshift;
		int flag_sfr;
		int flag_feedback;
		unsigned int npartTotal[6];
		int flag_cooling;
		int num_files;
		double BoxSize;
		double Omega0;
		double OmegaLambda;
		double HubbleParam;
		int flag_stellarage;
		int flag_metals;
		unsigned int npartTotalHighWord[6];
		int flag_entropy_instead_u;
		int flag_doubleprecision;
		char fill[empty_fill_bytes];
	};

protected:
	int num_files_;
	header this_header_;
	real_t lunit_, vunit_;

public:
	//! constructor
	explicit gadget2_output_interface(ConfigFile &cf )
	: output_interface(cf, "GADGET-2")
	{
		num_files_ = 1;
#ifdef USE_MPI
		// use as many output files as we have MPI tasks
		MPI_Comm_size(MPI_COMM_WORLD, &num_files_);
#endif
		real_t astart = 1.0/(1.0+cf_.GetValue<double>("setup", "zstart"));
		lunit_ = cf_.GetValue<double>("setup", "BoxLength");
		vunit_ = lunit_ / std::sqrt(astart);
	}

	real_t position_unit() const { return lunit_; }

	real_t velocity_unit() const { return vunit_; }

	void write_particle_data(const particle_container &pc)
	{
			// fill the Gadget-2 header
		memset(reinterpret_cast<void*>(&this_header_),0,sizeof(header));

		for (int i = 0; i < 6; ++i)
		{
			this_header_.npart[i] = 0;
			this_header_.npartTotal[i] = 0;
			this_header_.npartTotalHighWord[i] = 0;
		}

		this_header_.npart[1] = (pc.get_local_num_particles());
		this_header_.npartTotal[1] = (uint32_t)(pc.get_global_num_particles());
		this_header_.npartTotalHighWord[1] = (uint32_t)((pc.get_global_num_particles()) >> 32);

		/////
		//... set time ......................................................
		this_header_.redshift = cf_.GetValue<double>("setup", "zstart");
		this_header_.time = 1.0 / (1.0 + this_header_.redshift);

		//... SF flags
		this_header_.flag_sfr = 0;
		this_header_.flag_feedback = 0;
		this_header_.flag_cooling = 0;

		//...
		this_header_.num_files = num_files_; //1;
		this_header_.BoxSize = cf_.GetValue<double>("setup", "BoxLength");
		this_header_.Omega0 = cf_.GetValue<double>("cosmology", "Omega_m");
		this_header_.OmegaLambda = cf_.GetValue<double>("cosmology", "Omega_L");
		this_header_.HubbleParam = cf_.GetValue<double>("cosmology", "H0") / 100.0;

		this_header_.flag_stellarage = 0;
		this_header_.flag_metals = 0;

		this_header_.flag_entropy_instead_u = 0;

		// default units are in Mpc/h
		//if( kpcunits_ )
		//  this_header_.BoxSize *= 1000.0;
		// this_header_.BoxSize /= unit_length_chosen_;

		//... set masses
		double rhoc = 27.7519737; // in h^2 1e10 M_sol / Mpc^3
		double boxmass = this_header_.Omega0 * rhoc * std::pow(this_header_.BoxSize,3);
		this_header_.mass[1] = boxmass / pc.get_global_num_particles();
	
		std::string fname = fname_;
		int thisrank = 0;
		
#ifdef USE_MPI
		MPI_Comm_rank(MPI_COMM_WORLD,&thisrank);
		if( num_files_ > 1 )
			fname += "." + std::to_string(thisrank);
#endif
		uint32_t blocksz;
		std::ofstream ofs(fname.c_str(), std::ios::binary);

		csoca::ilog << "Writer \'" << this->interface_name_ << "\' : Writing data for " << pc.get_global_num_particles() << " particles." << std::endl;

		blocksz = sizeof(header);
		ofs.write( reinterpret_cast<char*>(&blocksz), sizeof(uint32_t) );
		ofs.write( reinterpret_cast<char*>(&this_header_), sizeof(header) );
		ofs.write( reinterpret_cast<char*>(&blocksz), sizeof(uint32_t) );
		
		blocksz = 3 * sizeof(float) * pc.get_local_num_particles();
		ofs.write( reinterpret_cast<char*>(&blocksz), sizeof(uint32_t) );
		ofs.write( reinterpret_cast<const char*>(pc.get_pos_ptr()), blocksz );
		ofs.write( reinterpret_cast<char*>(&blocksz), sizeof(uint32_t) );
		
		ofs.write( reinterpret_cast<char*>(&blocksz), sizeof(uint32_t) );
		ofs.write( reinterpret_cast<const char*>(pc.get_vel_ptr()), blocksz );
		ofs.write( reinterpret_cast<char*>(&blocksz), sizeof(uint32_t) );
		
		blocksz = sizeof(float) * pc.get_local_num_particles();
		ofs.write( reinterpret_cast<char*>(&blocksz), sizeof(uint32_t) );
		ofs.write( reinterpret_cast<const char*>(pc.get_ids_ptr()), blocksz );
		ofs.write( reinterpret_cast<char*>(&blocksz), sizeof(uint32_t) );
		
	}
};

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

#endif // __OUTPUT_HH
