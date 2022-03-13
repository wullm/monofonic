
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
#ifdef USE_HDF5
#include <unistd.h> // for unlink
#include <output_plugin.hh>
#include "HDF_IO.hh"

template <typename T>
std::vector<T> from_6array(const T *a)
{
  return std::vector<T>{{a[0], a[1], a[2], a[3], a[4], a[5]}};
}

template <typename T>
std::vector<T> from_value(const T a)
{
  return std::vector<T>{{a}};
}

template <typename write_real_t>
class gadget_hdf5_output_plugin : public output_plugin
{
  struct header_t
  {
    unsigned npart[6];
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
  };

protected:
  int num_files_, num_simultaneous_writers_;
  header_t header_;
  real_t lunit_, vunit_, munit_;
  bool blongids_;
  std::string this_fname_;

public:
  //! constructor
  explicit gadget_hdf5_output_plugin(config_file &cf, std::unique_ptr<cosmology::calculator> &pcc)
      : output_plugin(cf, pcc, "GADGET-HDF5")
  {
    num_files_ = 1;
#ifdef USE_MPI
    // use as many output files as we have MPI tasks
    MPI_Comm_size(MPI_COMM_WORLD, &num_files_);
#endif
    real_t astart = 1.0 / (1.0 + cf_.get_value<double>("setup", "zstart"));
    const double rhoc = 27.7519737; // in h^2 1e10 M_sol / Mpc^3

    lunit_ = cf_.get_value<double>("setup", "BoxLength");
    vunit_ = lunit_ / std::sqrt(astart);
    munit_ = rhoc * std::pow(cf_.get_value<double>("setup", "BoxLength"), 3); // in 1e10 h^-1 M_sol

    blongids_ = cf_.get_value_safe<bool>("output", "UseLongids", false);
    num_simultaneous_writers_ = cf_.get_value_safe<int>("output", "NumSimWriters", num_files_);

    for (int i = 0; i < 6; ++i)
    {
      header_.npart[i] = 0;
      header_.npartTotal[i] = 0;
      header_.npartTotalHighWord[i] = 0;
      header_.mass[i] = 0.0;
    }

    header_.time = astart;
    header_.redshift = 1.0 / astart - 1.0;
    header_.flag_sfr = 0;
    header_.flag_feedback = 0;
    header_.flag_cooling = 0;
    header_.num_files = num_files_;
    header_.BoxSize = lunit_;
    header_.Omega0 = pcc->cosmo_param_["Omega_m"];
    header_.OmegaLambda = pcc->cosmo_param_["Omega_DE"];
    header_.HubbleParam = pcc->cosmo_param_["h"];
    header_.flag_stellarage = 0;
    header_.flag_metals = 0;
    header_.flag_entropy_instead_u = 0;
    header_.flag_doubleprecision = (typeid(write_real_t) == typeid(double)) ? true : false;

    this_fname_ = fname_;
#ifdef USE_MPI
    int thisrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
    if (num_files_ > 1)
      this_fname_ += "." + std::to_string(thisrank);
#endif

    unlink(this_fname_.c_str());
    HDFCreateFile(this_fname_);
  }

  // use destructor to write header post factum
  ~gadget_hdf5_output_plugin()
  {
    if (!std::uncaught_exception()) 
    {   
      HDFCreateGroup(this_fname_, "Header");
      HDFWriteGroupAttribute(this_fname_, "Header", "NumPart_ThisFile", from_6array<unsigned>(header_.npart));
      HDFWriteGroupAttribute(this_fname_, "Header", "MassTable", from_6array<double>(header_.mass));
      HDFWriteGroupAttribute(this_fname_, "Header", "Time", from_value<double>(header_.time));
      HDFWriteGroupAttribute(this_fname_, "Header", "Redshift", from_value<double>(header_.redshift));
      HDFWriteGroupAttribute(this_fname_, "Header", "Flag_Sfr", from_value<int>(header_.flag_sfr));
      HDFWriteGroupAttribute(this_fname_, "Header", "Flag_Feedback", from_value<int>(header_.flag_feedback));
      HDFWriteGroupAttribute(this_fname_, "Header", "NumPart_Total", from_6array<unsigned>(header_.npartTotal));
      HDFWriteGroupAttribute(this_fname_, "Header", "Flag_Cooling", from_value<int>(header_.flag_cooling));
      HDFWriteGroupAttribute(this_fname_, "Header", "NumFilesPerSnapshot", from_value<int>(header_.num_files));
      HDFWriteGroupAttribute(this_fname_, "Header", "BoxSize", from_value<double>(header_.BoxSize));
      HDFWriteGroupAttribute(this_fname_, "Header", "Omega0", from_value<double>(header_.Omega0));
      HDFWriteGroupAttribute(this_fname_, "Header", "OmegaLambda", from_value<double>(header_.OmegaLambda));
      HDFWriteGroupAttribute(this_fname_, "Header", "HubbleParam", from_value<double>(header_.HubbleParam));
      HDFWriteGroupAttribute(this_fname_, "Header", "Flag_StellarAge", from_value<int>(header_.flag_stellarage));
      HDFWriteGroupAttribute(this_fname_, "Header", "Flag_Metals", from_value<int>(header_.flag_metals));
      HDFWriteGroupAttribute(this_fname_, "Header", "NumPart_Total_HighWord", from_6array<unsigned>(header_.npartTotalHighWord));
      HDFWriteGroupAttribute(this_fname_, "Header", "Flag_Entropy_ICs", from_value<int>(header_.flag_entropy_instead_u));

      music::ilog << "Wrote Gadget-HDF5 file(s) to " << this_fname_ << std::endl;
    }
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

  int get_species_idx(const cosmo_species &s) const
  {
    switch (s)
    {
    case cosmo_species::dm:
      return 1;
    case cosmo_species::baryon:
      return 0;
    case cosmo_species::neutrino:
      return 2;
    }
    return -1;
  }

  void set_particle_attributes(uint64_t numpart_local, uint64_t numpart_total, const cosmo_species &s, double Omega_species ) {
      int sid = get_species_idx(s);
      assert(sid != -1);

      header_.npart[sid] = numpart_local;
      header_.npartTotal[sid] = (uint32_t)(numpart_total);
      header_.npartTotalHighWord[sid] = (uint32_t)(numpart_total >> 32);

      if( header_.mass[get_species_idx(cosmo_species::dm)] == 0.0 )
        header_.mass[sid] = 0.0;
      else
        header_.mass[sid] = Omega_species * munit_ / numpart_total;
  };

  void write_particle_data(const particle::container &pc, const cosmo_species &s, double Omega_species)
  {
    int sid = get_species_idx(s);

    assert(sid != -1);

    header_.npart[sid] = (pc.get_local_num_particles());
    header_.npartTotal[sid] = (uint32_t)(pc.get_global_num_particles());
    header_.npartTotalHighWord[sid] = (uint32_t)((pc.get_global_num_particles()) >> 32);

    if( pc.bhas_individual_masses_ )
      header_.mass[sid] = 0.0;
    else
      header_.mass[sid] = Omega_species * munit_ / pc.get_global_num_particles();

    HDFCreateGroup(this_fname_, std::string("PartType") + std::to_string(sid));

    //... write positions and velocities.....
    if (this->has_64bit_reals())
    {
      HDFWriteDatasetVector(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/Coordinates"), pc.positions64_);
      HDFWriteDatasetVector(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/Velocities"), pc.velocities64_);
    }
    else
    {
      HDFWriteDatasetVector(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/Coordinates"), pc.positions32_);
      HDFWriteDatasetVector(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/Velocities"), pc.velocities32_);
    }

    //... write ids.....
    if (this->has_64bit_ids())
      HDFWriteDataset(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/ParticleIDs"), pc.ids64_);
    else
      HDFWriteDataset(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/ParticleIDs"), pc.ids32_);

    //... write masses.....
    if( pc.bhas_individual_masses_ ){
      if (this->has_64bit_reals()){
        HDFWriteDataset(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/Masses"), pc.mass64_);
      }else{
        HDFWriteDataset(this_fname_, std::string("PartType") + std::to_string(sid) + std::string("/Masses"), pc.mass32_);
      }
    }

    // std::cout << ">>>A> " << header_.npart[sid] << std::endl;
  }
};

namespace
{
output_plugin_creator_concrete<gadget_hdf5_output_plugin<float>> creator1("gadget_hdf5");
#if !defined(USE_SINGLEPRECISION)
output_plugin_creator_concrete<gadget_hdf5_output_plugin<double>> creator3("gadget_hdf5_double");
#endif
} // namespace

#endif