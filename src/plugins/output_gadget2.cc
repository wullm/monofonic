#include <fstream>
#include <output_plugin.hh>

constexpr int empty_fill_bytes{56};

template <typename write_real_t>
class gadget2_output_plugin : public output_plugin
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
	real_t lunit_, vunit_, munit_;
	bool blongids_;

public:
	//! constructor
	explicit gadget2_output_plugin(config_file &cf)
			: output_plugin(cf, "GADGET-2")
	{
		num_files_ = 1;
#ifdef USE_MPI
		// use as many output files as we have MPI tasks
		MPI_Comm_size(MPI_COMM_WORLD, &num_files_);
#endif
		const double rhoc = 27.7519737; // in h^2 1e10 M_sol / Mpc^3
		real_t astart = 1.0 / (1.0 + cf_.get_value<double>("setup", "zstart"));
		lunit_ = cf_.get_value<double>("setup", "BoxLength");
		vunit_ = lunit_ / std::sqrt(astart);
		munit_ = rhoc * std::pow(cf_.get_value<double>("setup", "BoxLength"), 3);;
		blongids_ = cf_.get_value_safe<bool>("output", "UseLongids", false);
	}

	output_type write_species_as(const cosmo_species &) const { return output_type::particles; }

	real_t position_unit() const { return lunit_; }

	real_t velocity_unit() const { return vunit_; }

	real_t mass_unit() const { return munit_; }

	bool has_64bit_reals() const
	{
		if (typeid(write_real_t) == typeid(double))
			return true;
		return false;
	}

	bool has_64bit_ids() const
	{
		if (blongids_)
			return true;
		return false;
	}

	void write_particle_data(const particle::container &pc, const cosmo_species &s, double Omega_species)
	{
		// fill the Gadget-2 header
		memset(reinterpret_cast<void *>(&this_header_), 0, sizeof(header));

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
		this_header_.redshift = cf_.get_value<double>("setup", "zstart");
		this_header_.time = 1.0 / (1.0 + this_header_.redshift);

		//... SF flags
		this_header_.flag_sfr = 0;
		this_header_.flag_feedback = 0;
		this_header_.flag_cooling = 0;

		//...
		this_header_.num_files = num_files_; //1;
		this_header_.BoxSize = cf_.get_value<double>("setup", "BoxLength");
		this_header_.Omega0 = cf_.get_value<double>("cosmology", "Omega_m");
		this_header_.OmegaLambda = cf_.get_value<double>("cosmology", "Omega_L");
		this_header_.HubbleParam = cf_.get_value<double>("cosmology", "H0") / 100.0;

		this_header_.flag_stellarage = 0;
		this_header_.flag_metals = 0;

		this_header_.flag_entropy_instead_u = 0;

		// default units are in Mpc/h
		//if( kpcunits_ )
		//  this_header_.BoxSize *= 1000.0;
		// this_header_.BoxSize /= unit_length_chosen_;

		//... set masses
		double rhoc = 27.7519737; // in h^2 1e10 M_sol / Mpc^3
		double boxmass = Omega_species * rhoc * std::pow(this_header_.BoxSize, 3);
		this_header_.mass[1] = boxmass / pc.get_global_num_particles();

		std::string fname = fname_;
		int thisrank = 0;

#ifdef USE_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
		if (num_files_ > 1)
			fname += "." + std::to_string(thisrank);
#endif
		uint32_t blocksz;
		std::ofstream ofs(fname.c_str(), std::ios::binary);

		music::ilog << "Writer \'" << this->interface_name_ << "\' : Writing data for " << pc.get_global_num_particles() << " particles." << std::endl;

		blocksz = sizeof(header);
		ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));
		ofs.write(reinterpret_cast<char *>(&this_header_), sizeof(header));
		ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));

		// we write double precision
		if (this->has_64bit_reals())
		{
			blocksz = 3 * sizeof(double) * pc.get_local_num_particles();
			ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));
			ofs.write(reinterpret_cast<const char *>(pc.get_pos64_ptr()), blocksz);
			ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));

			ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));
			ofs.write(reinterpret_cast<const char *>(pc.get_vel64_ptr()), blocksz);
			ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));
		}
		else
		{
			blocksz = 3 * sizeof(float) * pc.get_local_num_particles();
			ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));
			ofs.write(reinterpret_cast<const char *>(pc.get_pos32_ptr()), blocksz);
			ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));

			ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));
			ofs.write(reinterpret_cast<const char *>(pc.get_vel32_ptr()), blocksz);
			ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));
		}

		// we write long IDs
		if (this->has_64bit_ids())
		{
			blocksz = sizeof(uint64_t) * pc.get_local_num_particles();
			ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));
			ofs.write(reinterpret_cast<const char *>(pc.get_ids64_ptr()), blocksz);
			ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));
		}
		else
		{
			blocksz = sizeof(uint32_t) * pc.get_local_num_particles();
			ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));
			ofs.write(reinterpret_cast<const char *>(pc.get_ids32_ptr()), blocksz);
			ofs.write(reinterpret_cast<char *>(&blocksz), sizeof(uint32_t));
		}
	}
};

namespace
{
output_plugin_creator_concrete<gadget2_output_plugin<float>> creator1("gadget2");
#if !defined(USE_SINGLEPRECISION)
output_plugin_creator_concrete<gadget2_output_plugin<double>> creator3("gadget2_double");
#endif
} // namespace
