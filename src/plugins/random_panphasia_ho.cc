// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2021 by Oliver Hahn and Adrian Jenkins (this file)
// but see distinct licensing for PANPHASIA below
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
//
// IMPORTANT NOTICE:
// Note that PANPHASIA itself is not released under the GPL. Make sure
// to read and agree to its distinct licensing before you use or modify
// the code below or in the /external/panphasia directory which can be
// found here: http://icc.dur.ac.uk/Panphasia.php
// NOTE THAT PANPHASIA REQUIRES REGISTRATION ON THIS WEBSITE PRIOR TO USE

#if defined(USE_PANPHASIA_HO)

#include <general.hh>
#include <random_plugin.hh>
#include <config_file.hh>

#include <vector>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <grid_fft.hh>

extern "C"
{
  #include <panphasia_ho/panphasia_functions.h>
  extern size_t descriptor_base_size;
}

class RNG_panphasia_ho : public RNG_plugin
{
private:
protected:
  std::string descriptor_string_;
  int num_threads_;
  int panphasia_mode_;
  size_t grid_res_;

public:
  explicit RNG_panphasia_ho(config_file &cf) : RNG_plugin(cf)
  {

#ifdef _OPENMP
    num_threads_ = omp_get_max_threads();
#else
    num_threads_ = 1;
#endif

    descriptor_string_ = pcf_->get_value<std::string>("random", "descriptor");
    grid_res_ = pcf_->get_value<size_t>("setup", "GridRes");

    panphasia_mode_ = 0;
    parse_and_validate_descriptor_(descriptor_string_.c_str(), &panphasia_mode_);

    if (panphasia_mode_ == 0)
    {
      std::cout << "PANPHASIA: Old descriptor" << std::endl;
    }
    else if (panphasia_mode_ == 1)
    {
      std::cout << "PANPHASIA: New descriptor" << std::endl;

      int verbose = 0;
      int error;
      size_t x0 = 0, y0 = 0, z0 = 0;
      size_t rel_level;
      int fdim = 1; //Option to scale Fourier grid dimension relative to Panphasia coefficient grid

      //char descriptor[300] = "[Panph6,L20,(424060,82570,148256),S1,KK0,CH-999,Auriga_100_vol2]";

      PANPHASIA_init_descriptor_(descriptor_string_.c_str(), &verbose);

      printf("Descriptor %s\n ngrid_load %lu\n", descriptor_string_.c_str(), grid_res_);

      // Choose smallest value of level to equal of exceed grid_res_)

      for (rel_level = 0; fdim * (descriptor_base_size << (rel_level + 1)) <= grid_res_; rel_level++)
        ;

      printf("Setting relative level = %lu\n", rel_level);

      if ((error = PANPHASIA_init_level_(&rel_level, &x0, &y0, &z0, &verbose)))
      {
        printf("Abort: PANPHASIA_init_level_ :error code %d\n", error);
        abort();
      };

      //======================= FFTW ==============================

      ptrdiff_t alloc_local, local_n0, local_0_start;

      ptrdiff_t N0 = fdim * (descriptor_base_size << rel_level);

      alloc_local = FFTW_MPI_LOCAL_SIZE_3D(N0, N0, N0 + 2, MPI_COMM_WORLD, &local_n0, &local_0_start);

      FFTW_COMPLEX *Panphasia_White_Noise_Field;

      Panphasia_White_Noise_Field = FFTW_ALLOC_COMPLEX(alloc_local);

      if ((error = PANPHASIA_compute_kspace_field_(rel_level, N0, local_n0, local_0_start, Panphasia_White_Noise_Field)))
      {
        printf("Error code from PANPHASIA_compute ...  %d\n", error);
      };

      fftw_free(Panphasia_White_Noise_Field);

      //   PANPHASIA_HO_main(descriptor_string_.c_str(),&grid_res_);
    }
    else
    {
      std::cout << "PANPHASIA: Something went wrong with descriptor" << std::endl;
      abort();
    }
  }

  ~RNG_panphasia_ho()
  {
    if (panphasia_mode_ == 0) // old
    {
    }
  }

  bool isMultiscale() const { return true; }

  void Fill_Grid(Grid_FFT<real_t> &g)
  {
  }
};

namespace
{
  RNG_plugin_creator_concrete<RNG_panphasia_ho> creator("PANPHASIA_HO");
}
#endif // defined(USE_PANPHASIA_HO)
