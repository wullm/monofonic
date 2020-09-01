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

#ifdef USE_CLASS

#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <sstream>

#include <ClassEngine.hh>

#include <general.hh>
#include <config_file.hh>
#include <transfer_function_plugin.hh>
#include <math/interpolate.hh>

class transfer_CLASS_plugin : public TransferFunction_plugin
{

private:
  interpolated_function_1d<true, true, false> delta_c_, delta_b_, delta_n_, delta_m_, theta_c_, theta_b_, theta_n_, theta_m_;
  interpolated_function_1d<true, true, false> delta_c0_, delta_b0_, delta_n0_, delta_m0_, theta_c0_, theta_b0_, theta_n0_, theta_m0_;

  // single fluid growing/decaying mode decomposition
  // gsl_interp_accel *gsl_ia_Cplus_, *gsl_ia_Cminus_;
  // gsl_spline *gsl_sp_Cplus_, *gsl_sp_Cminus_;
  // std::vector<double> tab_Cplus_, tab_Cminus_;

  double Omega_m_, Omega_b_, N_ur_, zstart_, ztarget_, kmax_, kmin_, h_, astart_, atarget_, A_s_, n_s_, sigma8_, Tcmb_, tnorm_;

  ClassParams pars_;
  std::unique_ptr<ClassEngine> the_ClassEngine_;
  std::ofstream ofs_class_input_;

  template <typename T>
  void add_class_parameter(std::string parameter_name, const T parameter_value)
  {
    pars_.add(parameter_name, parameter_value);
    ofs_class_input_ << parameter_name << " = " << parameter_value << std::endl;
  }

  //! Set up class parameters from MUSIC cosmological parameters
  void init_ClassEngine(void)
  {
    //--- general parameters ------------------------------------------
    add_class_parameter("z_max_pk", std::max(std::max(zstart_, ztarget_),199.0)); // use 1.2 as safety
    add_class_parameter("P_k_max_h/Mpc", kmax_);
    add_class_parameter("output", "dTk,vTk");
    add_class_parameter("extra metric transfer functions","yes");
    // add_class_parameter("lensing", "no");

    //--- choose gauge ------------------------------------------------
    // add_class_parameter("extra metric transfer functions", "yes");
    add_class_parameter("gauge", "synchronous");

    //--- cosmological parameters, densities --------------------------
    add_class_parameter("h", h_);

    add_class_parameter("Omega_b", Omega_b_);
    add_class_parameter("Omega_cdm", Omega_m_ - Omega_b_);
    add_class_parameter("Omega_k", 0.0);
    add_class_parameter("Omega_fld", 0.0);
    add_class_parameter("Omega_scf", 0.0);


    // add_class_parameter("fluid_equation_of_state","CLP");
    // add_class_parameter("w0_fld", -1 );
    // add_class_parameter("wa_fld", 0. );
    // add_class_parameter("cs2_fld", 1);

    //--- massive neutrinos -------------------------------------------
#if 1
    //default off
    // add_class_parameter("Omega_ur",0.0);
    add_class_parameter("N_eff", N_ur_);
    add_class_parameter("N_ncdm", 0);

#else
    // change above to enable
    add_class_parameter("N_ur", 0);
    add_class_parameter("N_ncdm", 1);
    add_class_parameter("m_ncdm", "0.4");
    add_class_parameter("T_ncdm", 0.71611);
#endif

    //--- cosmological parameters, primordial -------------------------
    add_class_parameter("P_k_ini type", "analytic_Pk");

    if( A_s_ > 0.0 ){
      add_class_parameter("A_s", A_s_);
    }else{
      add_class_parameter("sigma8", sigma8_);
    }
    add_class_parameter("n_s", n_s_);
    add_class_parameter("alpha_s", 0.0);
    add_class_parameter("T_cmb", Tcmb_);
    add_class_parameter("YHe", 0.248);

    // additional parameters
    add_class_parameter("reio_parametrization", "reio_none");

    // precision parameters
    add_class_parameter("k_per_decade_for_pk", 100);
    add_class_parameter("k_per_decade_for_bao", 100);
    add_class_parameter("compute damping scale", "yes");
    add_class_parameter("tol_perturb_integration", 1.e-8);
    add_class_parameter("tol_background_integration", 1e-9);

    // high precision options from cl_permille.pre:
    // precision file to be passed as input in order to achieve at least percent precision on scalar Cls
    add_class_parameter("hyper_flat_approximation_nu", 7000.);
    add_class_parameter("transfer_neglect_delta_k_S_t0", 0.17);
    add_class_parameter("transfer_neglect_delta_k_S_t1", 0.05);
    add_class_parameter("transfer_neglect_delta_k_S_t2", 0.17);
    add_class_parameter("transfer_neglect_delta_k_S_e", 0.13);
    add_class_parameter("delta_l_max", 1000);

    int class_verbosity = 0;

    add_class_parameter("background_verbose", class_verbosity);
    add_class_parameter("thermodynamics_verbose", class_verbosity);
    add_class_parameter("perturbations_verbose", class_verbosity);
    add_class_parameter("transfer_verbose", class_verbosity);
    add_class_parameter("primordial_verbose", class_verbosity);
    add_class_parameter("spectra_verbose", class_verbosity);
    add_class_parameter("nonlinear_verbose", class_verbosity);
    add_class_parameter("lensing_verbose", class_verbosity);
    add_class_parameter("output_verbose", class_verbosity);

    // output parameters, only needed for the control CLASS .ini file that we output
    std::stringstream zlist;
    if (ztarget_ == zstart_)
      zlist << ztarget_ << ((ztarget_!=0.0)? ", 0.0" : "");
    else
      zlist << std::max(ztarget_, zstart_) << ", " << std::min(ztarget_, zstart_) << ", 0.0";
    add_class_parameter("z_pk", zlist.str());

    music::ilog << "Computing transfer function via ClassEngine..." << std::endl;
    double wtime = get_wtime();

    the_ClassEngine_ = std::move(std::make_unique<ClassEngine>(pars_, false));

    wtime = get_wtime() - wtime;
    music::ilog << "CLASS took " << wtime << " s." << std::endl;
  }

  //! run ClassEngine with parameters set up
  void run_ClassEngine(double z, std::vector<double> &k, std::vector<double> &dc, std::vector<double> &tc, std::vector<double> &db, std::vector<double> &tb,
                       std::vector<double> &dn, std::vector<double> &tn, std::vector<double> &dm, std::vector<double> &tm)
  {
    k.clear(); 
    dc.clear(); db.clear(); dn.clear(); dm.clear();
    tc.clear(); tb.clear(); tn.clear(); tm.clear();
    
    the_ClassEngine_->getTk(z, k, dc, db, dn, dm, tc, tb, tn, tm);

    real_t fc = (Omega_m_ - Omega_b_) / Omega_m_;
    real_t fb = Omega_b_ / Omega_m_;

    for (size_t i = 0; i < k.size(); ++i)
    {
      // convert to 'CAMB' format, since we interpolate loglog and
      // don't want negative numbers...
      auto ik2 = 1.0 / (k[i] * k[i]) * h_ * h_;
      dc[i] = -dc[i] * ik2;
      db[i] = -db[i] * ik2;
      dn[i] = -dn[i] * ik2;
      dm[i] = fc * dc[i] + fb * db[i];
      tc[i] = -tc[i] * ik2;
      tb[i] = -tb[i] * ik2;
      tn[i] = -tn[i] * ik2;
      tm[i] = fc * tc[i] + fb * tb[i];
    }
  }

public:
  explicit transfer_CLASS_plugin(config_file &cf)
      : TransferFunction_plugin(cf)
  {
    this->tf_isnormalised_ = true;

    ofs_class_input_.open("input_class_parameters.ini", std::ios::trunc);

    h_ = pcf_->get_value<double>("cosmology", "H0") / 100.0;
    Omega_m_ = pcf_->get_value<double>("cosmology", "Omega_m");
    Omega_b_ = pcf_->get_value<double>("cosmology", "Omega_b");
    N_ur_ = pcf_->get_value_safe<double>("cosmology", "Neff", 3.046);
    ztarget_ = pcf_->get_value_safe<double>("cosmology", "ztarget", 0.0);
    atarget_ = 1.0 / (1.0 + ztarget_);
    zstart_ = pcf_->get_value<double>("setup", "zstart");
    astart_ = 1.0 / (1.0 + zstart_);
    A_s_ = pcf_->get_value_safe<double>("cosmology", "A_s", -1.0);
    n_s_ = pcf_->get_value<double>("cosmology", "nspec");
    Tcmb_ = cf.get_value_safe<double>("cosmology", "Tcmb", 2.7255);

    if (A_s_ > 0) {
      music::ilog << "CLASS: Using A_s=" << A_s_<< " to normalise the transfer function." << std::endl;
    }else{
      sigma8_ = pcf_->get_value_safe<double>("cosmology", "sigma_8", -1.0);
      if( sigma8_ < 0 ){
        throw std::runtime_error("Need to specify either A_s or sigma_8 for CLASS plugin...");
      }
      music::ilog << "CLASS: Using sigma8_ =" << sigma8_<< " to normalise the transfer function." << std::endl;
    }

    // determine highest k we will need for the resolution selected
    double lbox = pcf_->get_value<double>("setup", "BoxLength");
    int nres = pcf_->get_value<double>("setup", "GridRes");
    kmax_ = std::max(20.0, 2.0 * M_PI / lbox * nres / 2 * sqrt(3) * 2.0); // 120% of spatial diagonal, or k=10h Mpc-1

    // initialise CLASS and get the normalisation
    this->init_ClassEngine();
    A_s_ = the_ClassEngine_->get_A_s(); // this either the input one, or the one computed from sigma8
    
    // compute the normalisation to interface with MUSIC
    double k_p = pcf_->get_value_safe<double>("cosmology", "k_p", 0.05);
    tnorm_ = std::sqrt(2.0 * M_PI * M_PI * A_s_ * std::pow(1.0 / k_p * h_, n_s_ - 1) / std::pow(2.0 * M_PI, 3.0));

    // compute the transfer function at z=0 using CLASS engine
    std::vector<double> k, dc, tc, db, tb, dn, tn, dm, tm;
    this->run_ClassEngine(0.0, k, dc, tc, db, tb, dn, tn, dm, tm);

    delta_c0_.set_data(k, dc);
    theta_c0_.set_data(k, tc);
    delta_b0_.set_data(k, db);
    theta_b0_.set_data(k, tb);
    delta_n0_.set_data(k, dn);
    theta_n0_.set_data(k, tn);
    delta_m0_.set_data(k, dm);
    theta_m0_.set_data(k, tm);

     // compute the transfer function at z=z_target using CLASS engine
    this->run_ClassEngine(ztarget_, k, dc, tc, db, tb, dn, tn, dm, tm);
    delta_c_.set_data(k, dc);
    theta_c_.set_data(k, tc);
    delta_b_.set_data(k, db);
    theta_b_.set_data(k, tb);
    delta_n_.set_data(k, dn);
    theta_n_.set_data(k, tn);
    delta_m_.set_data(k, dm);
    theta_m_.set_data(k, tm);

    kmin_ = k[0];
    kmax_ = k.back();

    music::ilog << "CLASS table contains k = " << this->get_kmin() << " to " << this->get_kmax() << " h Mpc-1." << std::endl;

    tf_distinct_ = true;
    tf_withvel_ = true;
    tf_withtotal0_ = true;
  }

  ~transfer_CLASS_plugin()
  {
  }

  inline double compute(double k, tf_type type) const
  {
    k *= h_;

    if (k < kmin_ || k > kmax_)
    {
      return 0.0;
    }

    real_t val(0.0);
    switch (type)
    {
      // values at ztarget:
    case total:
      val = delta_m_(k); break;
    case cdm:
      val = delta_c_(k); break;
    case baryon:
      val = delta_b_(k); break;
    case vtotal:
      val = theta_m_(k); break;
    case vcdm:
      val = theta_c_(k); break;
    case vbaryon:
      val = theta_b_(k); break;
    case deltabc:
      val = delta_b_(k)-delta_c_(k); break;

      // values at zstart:
    case total0:
      val = delta_m0_(k); break;
    case cdm0:
      val = delta_c0_(k); break;
    case baryon0:
      val = delta_b0_(k); break;
    case vtotal0:
      val = theta_m0_(k); break;
    case vcdm0:
      val = theta_c0_(k); break;
    case vbaryon0:
      val = theta_b0_(k); break;
    default:
      throw std::runtime_error("Invalid type requested in transfer function evaluation");
    }
    return val * tnorm_;
  }

  inline double get_kmin(void) const { return kmin_ / h_; }
  inline double get_kmax(void) const { return kmax_ / h_; }
};

namespace
{
TransferFunction_plugin_creator_concrete<transfer_CLASS_plugin> creator("CLASS");
}

#endif // USE_CLASS
