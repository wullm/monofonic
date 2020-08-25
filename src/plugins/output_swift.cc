
// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2020 by Oliver Hahn & Michael Buehlmann (this file)
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
#include <array>
#include <output_plugin.hh>
#include "HDF_IO.hh"

template <typename T>
std::vector<T> from_6array(const std::array<T,6>& a)
{
  return std::vector<T>{{a[0], a[1], a[2], a[3], a[4], a[5]}};
}

template <typename T>
std::vector<T> from_value(const T a)
{
  return std::vector<T>{{a}};
}

template <typename write_real_t>
class swift_output_plugin : public output_plugin
{

protected:
  int num_files_, num_simultaneous_writers_;

  real_t lunit_, vunit_, munit_;
  real_t boxsize_, hubble_param_, astart_;
  bool blongids_, bdobaryons_;
  std::string this_fname_;

  std::array<uint32_t,6> npart_, npartTotal_, npartTotalHighWord_;
  std::array<double,6> mass_;
  double time_;

public:
  //! constructor
  explicit swift_output_plugin(config_file &cf)
      : output_plugin(cf, "SWIFT")
  {
    num_files_ = 1;
#ifdef USE_MPI
    // use as many output files as we have MPI tasks
    MPI_Comm_size(MPI_COMM_WORLD, &num_files_);
#endif
    real_t astart = 1.0 / (1.0 + cf_.get_value<double>("setup", "zstart"));
    const double rhoc = 27.7519737; // in h^2 1e10 M_sol / Mpc^3

    hubble_param_ = cf_.get_value<double>("cosmology", "H0") / 100.;
    astart_ = 1.0/(1.0+cf_.get_value<double>("setup","zstart"));
    boxsize_ = cf_.get_value<double>("setup", "BoxLength");

    lunit_ = boxsize_ / hubble_param_; // final units will be in Mpc (without h)
    vunit_ = boxsize_; // final units will be in km/s
    munit_ = rhoc * std::pow(boxsize_, 3) / hubble_param_; // final units will be in 1e10 M_sol    

    blongids_ = cf_.get_value_safe<bool>("output", "UseLongids", false);
    bdobaryons_ = cf_.get_value<bool>("setup","DoBaryons");

    for (int i = 0; i < 6; ++i)
    {
      npart_[i] = 0;
      npartTotal_[i] = 0;
      npartTotalHighWord_[i] = 0;
      mass_[i] = 0.0;
    }

    time_ = astart;
    
    this_fname_ = fname_;
#ifdef USE_MPI
    int thisrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
    if (num_files_ > 1)
      this_fname_ += "." + std::to_string(thisrank);
#endif

    // delete output file if it exists
    unlink(this_fname_.c_str());

    // create output HDF5 file
    HDFCreateFile(this_fname_);

    // Write UNITS header
    HDFCreateGroup(this_fname_, "Units");
    HDFWriteGroupAttribute(this_fname_, "Units", "Unit mass in cgs (U_M)", 1.98848e43);      // 10^10 Msun in grams
    HDFWriteGroupAttribute(this_fname_, "Units", "Unit length in cgs (U_L)", 3.08567758e24); // 1 Mpc in cm
    HDFWriteGroupAttribute(this_fname_, "Units", "Unit time in cgs (U_t)", 3.08567758e19);   // so that unit vel is 1 km/s
    HDFWriteGroupAttribute(this_fname_, "Units", "Unit current in cgs (U_I)", 1.0);            // 1 Ampere
    HDFWriteGroupAttribute(this_fname_, "Units", "Unit temperature in cgs (U_T)", 1.0);               // 1 Kelvin

    // TODO: Write MUSIC configuration header
    HDFCreateGroup(fname_, "ICs_parameters");
    // ...
  }

  // use destructor to write header post factum
  ~swift_output_plugin()
  {
    if (!std::uncaught_exception()) 
    {   
      HDFCreateGroup(this_fname_, "Header");
      HDFWriteGroupAttribute(fname_, "Header", "Dimension", 3);
      HDFWriteGroupAttribute(fname_, "Header", "BoxSize", lunit_);  // in Mpc, not Mpc/h
      
      HDFWriteGroupAttribute(this_fname_, "Header", "NumPart_Total", from_6array<unsigned>(npartTotal_));
      HDFWriteGroupAttribute(this_fname_, "Header", "NumPart_Total_HighWord", from_6array<unsigned>(npartTotalHighWord_));
      HDFWriteGroupAttribute(this_fname_, "Header", "NumPart_ThisFile", from_6array<unsigned>(npart_));
      HDFWriteGroupAttribute(this_fname_, "Header", "MassTable", from_6array<double>(mass_));
      
      HDFWriteGroupAttribute(this_fname_, "Header", "Time", from_value<double>(time_));
      HDFWriteGroupAttribute(this_fname_, "Header", "Flag_Entropy_ICs", from_value<int>(0));

      HDFWriteGroupAttribute(this_fname_, "Header", "NumFilesPerSnapshot", from_value<int>(num_files_));
      
      // write GAS internal energy if baryons are enabled
      if( bdobaryons_ )
      {
        const double gamma  = cf_.get_value_safe<double>("cosmology", "gamma", 5.0 / 3.0);
        const double YHe    = cf_.get_value_safe<double>("cosmology", "YHe", 0.248);
        //const double Omega0 = cf_.get_value<double>("cosmology", "Omega_m");
        const double omegab = cf_.get_value<double>("cosmology", "Omega_b");
        const double Tcmb0  = cf_.get_value_safe<double>("cosmology", "Tcmb", 2.7255);
        

        // compute gas internal energy
        const double npol = (fabs(1.0 - gamma) > 1e-7) ? 1.0 / (gamma - 1.) : 1.0;
        const double unitv = 1e5;
        const double adec = 1.0 / (160. * std::pow(omegab * hubble_param_ * hubble_param_ / 0.022, 2.0 / 5.0));
        const double Tini = astart_ < adec ? Tcmb0 / astart_ : Tcmb0 / astart_ / astart_ * adec;
        const double mu = (Tini > 1.e4) ? 4.0 / (8. - 5. * YHe) : 4.0 / (1. + 3. * (1. - YHe));
        const double ceint = 1.3806e-16 / 1.6726e-24 * Tini * npol / mu / unitv / unitv;

        music::ilog.Print("Swift : set initial gas temperature to %.2f K/mu", Tini / mu);

        std::vector<write_real_t> data( npart_[0], ceint );
        HDFWriteDataset(this_fname_, "PartType0/InternalEnergy", data);
      }

      music::ilog << "Wrote SWIFT IC file(s) to " << this_fname_ << std::endl;
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
      return 3;
    }
    return -1;
  }

  void write_particle_data(const particle::container &pc, const cosmo_species &s, double Omega_species)
  {
    int sid = get_species_idx(s);

    assert(sid != -1);

    npart_[sid] = (pc.get_local_num_particles());
    npartTotal_[sid] = (uint32_t)(pc.get_global_num_particles());
    npartTotalHighWord_[sid] = (uint32_t)((pc.get_global_num_particles()) >> 32);

    if( pc.bhas_individual_masses_ )
      mass_[sid] = 0.0;
    else
      mass_[sid] = Omega_species * munit_ / pc.get_global_num_particles();

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
  }
};

namespace
{
output_plugin_creator_concrete<swift_output_plugin<float>> creator1("SWIFT");
} // namespace

#endif
