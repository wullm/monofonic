// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2020 by Oliver Hahn
// Copyright (C) 2021 by Willem Elbers
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
#include <ic_generator.hh>

#include <math/interpolate.hh>

#include "../../external/3fa/cosmology_tables.h"
#include "../../external/3fa/fluid_equations.h"

class transfer_3FA_CLASS_plugin : public TransferFunction_plugin
{
private:
  
  using TransferFunction_plugin::cosmo_params_;

  interpolated_function_1d<true, true, false> delta_c_, delta_b_, delta_n_, delta_m_, theta_c_, theta_b_, theta_n_, theta_m_;
  interpolated_function_1d<true, true, false> delta_c0_, delta_b0_, delta_n0_, delta_m0_, theta_c0_, theta_b0_, theta_n0_, theta_m0_;

  double zstart_, ztarget_, astart_, atarget_, kmax_, kmin_, h_, tnorm_;
  
  // asymptotic growth factor and growth rates at large k
  double Dm_asymptotic_, fm_asymptotic_, vfac_asymptotic_;

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
    add_class_parameter("P_k_max_h/Mpc", std::max(2.0,kmax_));
    add_class_parameter("output", "dTk,vTk");
    add_class_parameter("extra metric transfer functions","yes");
    // add_class_parameter("lensing", "no");

    //--- choose gauge ------------------------------------------------
    // add_class_parameter("extra metric transfer functions", "yes");
    add_class_parameter("gauge", "synchronous");

    //--- cosmological parameters, densities --------------------------
    add_class_parameter("h", cosmo_params_.get("h"));

    add_class_parameter("Omega_b", cosmo_params_.get("Omega_b"));
    add_class_parameter("Omega_cdm", cosmo_params_.get("Omega_c"));
    add_class_parameter("Omega_k", cosmo_params_.get("Omega_k"));
    add_class_parameter("Omega_fld", 0.0);
    add_class_parameter("Omega_scf", 0.0);


    // add_class_parameter("fluid_equation_of_state","CLP");
    // add_class_parameter("w0_fld", -1 );
    // add_class_parameter("wa_fld", 0. );
    // add_class_parameter("cs2_fld", 1);

    //--- massive neutrinos -------------------------------------------
    add_class_parameter("N_ur", cosmo_params_.get("N_ur"));
    add_class_parameter("N_ncdm", cosmo_params_.get("N_nu_massive"));
    if( cosmo_params_.get("N_nu_massive") > 0 ){
      std::stringstream sstr;
      if( cosmo_params_.get("m_nu1") > 1e-9 ) sstr << cosmo_params_.get("m_nu1");
      if( cosmo_params_.get("m_nu2") > 1e-9 ) sstr << ", " << cosmo_params_.get("m_nu2");
      if( cosmo_params_.get("m_nu3") > 1e-9 ) sstr << ", " << cosmo_params_.get("m_nu3");
      add_class_parameter("m_ncdm", sstr.str().c_str());
    }
    
#endif

    //--- cosmological parameters, primordial -------------------------
    add_class_parameter("P_k_ini type", "analytic_Pk");

    if( cosmo_params_.get("A_s") > 0.0 ){
      add_class_parameter("A_s", cosmo_params_.get("A_s"));
    }else{
      add_class_parameter("sigma8", cosmo_params_.get("sigma_8"));
    }
    add_class_parameter("n_s", cosmo_params_.get("n_s"));
    add_class_parameter("alpha_s", 0.0);
    add_class_parameter("T_cmb", cosmo_params_.get("Tcmb"));
    add_class_parameter("YHe", cosmo_params_.get("YHe"));

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

    const double h  = cosmo_params_.get("h");

    for (size_t i = 0; i < k.size(); ++i)
    {
      // convert to 'CAMB' format, since we interpolate loglog and
      // don't want negative numbers...
      auto ik2 = 1.0 / (k[i] * k[i]) * h * h;
      dc[i] = -dc[i] * ik2;
      db[i] = -db[i] * ik2;
      dn[i] = -dn[i] * ik2;
      dm[i] = -dm[i] * ik2;
      tc[i] = -tc[i] * ik2;
      tb[i] = -tb[i] * ik2;
      tn[i] = -tn[i] * ik2;
      tm[i] = -tm[i] * ik2;
    }
  }

public:
  explicit transfer_3FA_CLASS_plugin(config_file &cf, const cosmology::parameters& cosmo_params)
      : TransferFunction_plugin(cf,cosmo_params)
  {
    this->tf_isnormalised_ = true;

    ofs_class_input_.open("input_class_parameters.ini", std::ios::trunc);

    // all cosmological parameters need to be passed through the_cosmo_calc
    
    ztarget_ = pcf_->get_value_safe<double>("cosmology", "ztarget", 0.0);
    atarget_ = 1.0 / (1.0 + ztarget_);
    zstart_ = pcf_->get_value<double>("setup", "zstart");
    astart_ = 1.0 / (1.0 + zstart_);

    h_ = cosmo_params_["h"];
    
    if (cosmo_params_["A_s"] > 0.0) {
      music::ilog << "CLASS: Using A_s=" << cosmo_params_["A_s"] << " to normalise the transfer function." << std::endl;
    }else{
      double sigma8 = cosmo_params_["sigma_8"];
      if( sigma8 < 0 ){
        throw std::runtime_error("Need to specify either A_s or sigma_8 for CLASS plugin...");
      }
      music::ilog << "CLASS: Using sigma8_ =" << sigma8<< " to normalise the transfer function." << std::endl;
    }

    // determine highest k we will need for the resolution selected
    double lbox = pcf_->get_value<double>("setup", "BoxLength");
    int nres = pcf_->get_value<double>("setup", "GridRes");
    kmax_ = std::max(20.0, 2.0 * M_PI / lbox * nres / 2 * sqrt(3) * 2.0); // 120% of spatial diagonal, or k=10h Mpc-1

    // initialise CLASS and get the normalisation
    this->init_ClassEngine();
    double A_s_ = the_ClassEngine_->get_A_s(); // this either the input one, or the one computed from sigma8
    
    // compute the normalisation to interface with MUSIC
    double k_p = cosmo_params["k_p"] / cosmo_params["h"];
    tnorm_ = std::sqrt(2.0 * M_PI * M_PI * A_s_ * std::pow(1.0 / k_p, cosmo_params["n_s"] - 1) / std::pow(2.0 * M_PI, 3.0));

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
    std::vector<double> dc_target, tc_target, db_target, tb_target, dn_target, tn_target, dm_target, tm_target;
    this->run_ClassEngine(ztarget_, k, dc_target, tc_target, db_target, tb_target, dn_target, tn_target, dm_target, tm_target);
    
    // compute the transfer function at z=z_start using CLASS engine
    this->run_ClassEngine(zstart_, k, dc, tc, db, tb, dn, tn, dm, tm);
    
    // evaluate transfer functions at a_min < a_start and a_plus > a_start
    const double delta_log_a = 0.002;
    const double log_astart_ = log(astart_);
    const double a_min = exp(log_astart_ - delta_log_a);
    const double a_pls = exp(log_astart_ + delta_log_a);
    const double z_min = 1.0 / a_min - 1.0;
    const double z_pls = 1.0 / a_pls - 1.0;

    std::vector<double> dc_min, tc_min, db_min, tb_min, dn_min, tn_min, dm_min, tm_min;
    std::vector<double> dc_pls, tc_pls, db_pls, tb_pls, dn_pls, tn_pls, dm_pls, tm_pls;
    
    // compute the transfer functions at z=z_min and z=z_plus using CLASS engine
    this->run_ClassEngine(z_min, k, dc_min, tc_min, db_min, tb_min, dn_min, tn_min, dm_min, tm_min);
    this->run_ClassEngine(z_pls, k, dc_pls, tc_pls, db_pls, tb_pls, dn_pls, tn_pls, dm_pls, tm_pls);
    
    // compute the logarithmic growth rates at z=z_start using CLASS engine
    std::vector<double> gc, gb, gn;
    for (size_t i = 0; i < k.size(); ++i)
    {
      gc.push_back((dc_pls[i] - dc_min[i]) / (2.0 * delta_log_a) / dc[i]);
      gb.push_back((db_pls[i] - db_min[i]) / (2.0 * delta_log_a) / db[i]);
      gn.push_back((dn_pls[i] - dn_min[i]) / (2.0 * delta_log_a) / dn[i]);
    }
    
    kmin_ = k[0];
    kmax_ = k.back();

    music::ilog << "CLASS table contains k = " << this->get_kmin() << " to " << this->get_kmax() << " h Mpc-1." << std::endl;
    
    // array of neutrino masses in eV needed by 3FA
    const int N_nu = cosmo_params_.get("N_nu_massive");
    std::vector<double> M_nu;
    std::vector<double> deg_nu; //degeneracies
    
    if( cosmo_params_.get("N_nu_massive") > 0 ){
        if( cosmo_params_.get("m_nu1") > 1e-9 ) {
            M_nu.push_back(cosmo_params_.get("m_nu1"));
            deg_nu.push_back(1.0);
        }
        if( cosmo_params_.get("m_nu2") > 1e-9 ) {
            M_nu.push_back(cosmo_params_.get("m_nu2"));
            deg_nu.push_back(1.0);
        }
        if( cosmo_params_.get("m_nu1") > 1e-9 ) {
            M_nu.push_back(cosmo_params_.get("m_nu3"));
            deg_nu.push_back(1.0);
        }
    }
  
    // Set up 3FA cosmological parameters
    struct model m;
    m.h = cosmo_params_.get("h");
    m.Omega_b = cosmo_params_.get("Omega_b");
    m.Omega_c = cosmo_params_.get("Omega_c");
    m.Omega_k = cosmo_params_.get("Omega_k");
    m.N_ur = cosmo_params_.get("N_ur");
    m.N_nu = N_nu;
    m.M_nu = M_nu.data();
    m.deg_nu = deg_nu.data();
    m.T_nu_0 = 1.951757805; //default CLASS value
    m.T_CMB_0 = cosmo_params_.get("Tcmb");
    m.w0 = -1.0;
    m.wa = 0.0;
    m.sim_neutrino_nonrel_masses = 1; //TODO: make parameter

    // Set up 3FA unit system
    struct units us;
    us.UnitLengthMetres = 3.085677581491e+022; //Mpc
    us.UnitTimeSeconds = 3.153600000000e+016; //Gyr
    us.UnitMassKilogram = 1.988435e40; //1e10 M_sol
    us.UnitTemperatureKelvin = 1.0;
    us.UnitCurrentAmpere = 1.0;
    set_physical_constants(&us);

    double wtime = get_wtime();
    music::ilog << "Integrating cosmological tables with 3FA." << std::endl;

    // Integrate the cosmological tables with 3FA (correctly accounting for neutrinos)
    struct cosmology_tables tab;
    integrate_cosmology_tables(&m, &us, &tab, 1000);

    // extract the present-day neutrino fraction and the baryon fraction
    const double f_nu_nr_0 = tab.f_nu_nr[tab.size-1];
    const double f_b = m.Omega_b / (m.Omega_b + m.Omega_c);
  
    music::ilog << "Integrating fluid equations with 3FA." << std::endl;

    // compute the scale-dependent growth factors in the 3-fluid approximation
    std::vector<double> Dc, Db, Dn;
    for (size_t i = 0; i < k.size(); ++i)
    {
        // initialise the input data for the fluid equations
        struct growth_factors gfac;
        gfac.k = k[i];
        gfac.Dc = dc[i];
        gfac.Db = db[i];
        gfac.Dn = dn[i];
        gfac.gc = gc[i];
        gfac.gb = gb[i];
        gfac.gn = gn[i];
    
        integrate_fluid_equations(&m, &us, &tab, &gfac, astart_, atarget_);
    
        // store the relative growth factors between the target and starting redshifts
        Dc.push_back(gfac.Dc);
        Db.push_back(gfac.Db);
        Dn.push_back(gfac.Dn); 
    }
  
    wtime = get_wtime() - wtime;
    music::ilog << "3FA took " << wtime << " s." << std::endl;

    // determine the asymptotic total matter growth rate and factor by averaging
    // over small scales modes (k > 1/Mpc)
    double Dm_sum = 0.;
    double gm_sum = 0.;
    int count = 0;
    for (size_t i = 0; i < k.size(); ++i)
    {
        if (k[i] < 1.0) continue; //ignore large scales
      
        double Dcb = f_b * Db[i] + (1-f_b) * Dc[i];
        double Dm = f_nu_nr_0 * Dn[i] + (1-f_nu_nr_0) * Dcb;
        double gcb = (f_b * Db[i] * gb[i] + (1-f_b) * Dc[i] * gc[i]) / Dcb;
        double gm = (f_nu_nr_0 * Dn[i] * gn[i] + (1-f_nu_nr_0) * Dcb * gcb) / Dm;
      
        Dm_sum += Dm;
        gm_sum += gm;
        count++;      
    }
  
    // normalize the Hubble rate at astart_
    real_t H_start = get_H_of_a(&tab, astart_) / get_H_of_a(&tab, 1.0) * cosmo_params_.get("H0");
    
    Dm_asymptotic_ = Dm_sum / count;
    fm_asymptotic_ = gm_sum / count;
    vfac_asymptotic_ = astart_ * H_start * fm_asymptotic_ / cosmo_params_.get("h");
  
    // now scale forward with the asymptotic growth factor, as assumed in the ic generator
    for (size_t i = 0; i < k.size(); ++i)
    {
        // scale back the density transfer functions from the target redshift
        // to the starting redshift  using the scale-dependent growth factors 
        dc[i] = dc_target[i] * Dc[i];
        db[i] = db_target[i] * Db[i];
        dn[i] = dn_target[i] * Dn[i];
        
        // scale the transfer functions forward with the total asymptotic factor
        dc[i] /= Dm_asymptotic_;
        db[i] /= Dm_asymptotic_;
        dn[i] /= Dm_asymptotic_;
        tc[i] /= Dm_asymptotic_;
        tb[i] /= Dm_asymptotic_;
        tn[i] /= Dm_asymptotic_;
      
        double dcb, tcb;
        
        // compute the mass-weighted average 
        dcb = f_b * dc[i] + (1.0 - f_b) * db[i];
        dm[i] = f_nu_nr_0 * dn[i] + (1-f_nu_nr_0) * dcb;
        tcb = f_b * tc[i] + (1.0 - f_b) * tb[i];
        tm[i] = f_nu_nr_0 * tn[i] + (1-f_nu_nr_0) * tcb;
      
        // use the compensated (baryon-cdm) modes from the target redshift
        db[i] = dcb - f_b * (db_target[i] - dc_target[i]);
        dc[i] = dcb + (1.0 - f_b) * (db_target[i] - dc_target[i]);
        tb[i] = tcb - f_b * (tb_target[i] - tc_target[i]);
        tc[i] = tcb + (1.0 - f_b) * (tb_target[i] - tc_target[i]);
    }
  
    // Store the rescaled transfer function data
    delta_c_.set_data(k, dc);
    delta_b_.set_data(k, db);
    delta_n_.set_data(k, dn);
    delta_m_.set_data(k, dm);
    theta_c_.set_data(k, dc);
    theta_b_.set_data(k, db);
    theta_n_.set_data(k, dn);
    theta_m_.set_data(k, dm);  
  
    music::ilog << "Asymptotic Dm = " << Dm_asymptotic_ << std::endl;
    music::ilog << "Asymptotic fm = " << fm_asymptotic_ << std::endl;
    music::ilog << "Asymptotic aHf/h = " << vfac_asymptotic_ << std::endl;
  
    // clean up 3FA
    free_cosmology_tables(&tab);
    
    tf_with_Dm_asymptotic_ = true;
    tf_distinct_ = true;
    tf_withvel_ = true;
    tf_withtotal0_ = true;
  }

  ~transfer_3FA_CLASS_plugin()
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
    case delta_matter:
      val = delta_m_(k); break;
    case delta_cdm:
      val = delta_c_(k); break;
    case delta_baryon:
      val = delta_b_(k); break;
    case theta_matter:
      val = theta_m_(k); break;
    case theta_cdm:
      val = theta_c_(k); break;
    case theta_baryon:
      val = theta_b_(k); break;
    case delta_bc:
      val = delta_b_(k)-delta_c_(k); break;
    case theta_bc:
      val = theta_b_(k)-theta_c_(k); break;
    case delta_nu:
      val = delta_n_(k); break;
    case theta_nu:
      val = theta_n_(k); break;

      // values at zstart:
    case delta_matter0:
      val = delta_m0_(k); break;
    case delta_cdm0:
      val = delta_c0_(k); break;
    case delta_baryon0:
      val = delta_b0_(k); break;
    case theta_matter0:
      val = theta_m0_(k); break;
    case theta_cdm0:
      val = theta_c0_(k); break;
    case theta_baryon0:
      val = theta_b0_(k); break;
    case delta_nu0:
      val = delta_n0_(k); break;
    case theta_nu0:
      val = theta_n0_(k); break;

    default:
      throw std::runtime_error("Invalid type requested in transfer function evaluation");
    }
    return val * tnorm_;
  }

  inline double get_kmin(void) const { return kmin_ / h_; }
  inline double get_kmax(void) const { return kmax_ / h_; }
  inline double get_Dm_asymptotic(void) const { return Dm_asymptotic_; }
  inline double get_fm_asymptotic(void) const { return fm_asymptotic_; }
  inline double get_vfac_asymptotic(void) const { return vfac_asymptotic_; }
};

namespace
{
TransferFunction_plugin_creator_concrete<transfer_3FA_CLASS_plugin> creator("3FA_CLASS");
}

#endif // USE_CLASS