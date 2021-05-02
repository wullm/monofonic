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

extern "C"{
  int PANPHASIA_HO_main( void );
}


class RNG_panphasia_ho : public RNG_plugin
{
private:
protected:
  std::string descriptor_string_;
  int num_threads_;
  

public:
  explicit RNG_panphasia_ho(config_file &cf) : RNG_plugin(cf)
  {
    descriptor_string_ = pcf_->get_value<std::string>("random", "descriptor");

#ifdef _OPENMP
    num_threads_ = omp_get_max_threads();
#else
    num_threads_ = 1;
#endif

    PANPHASIA_HO_main();
  }

  ~RNG_panphasia_ho() {  }

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