//  transfer_CLASS.cc - This file is part of MUSIC -
//  a code to generate multi-scale initial conditions for cosmological simulations

//  Copyright (C) 2019  Oliver Hahn


#include <cmath>
#include <string>
#include <vector>
#include <memory>

#include <ClassEngine.hh>

#include <general.hh>
#include <config_file.hh>
#include <transfer_function_plugin.hh>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

const double tiny = 1e-30;

class transfer_CLASS_plugin : public TransferFunction_plugin {

private:
    std::vector<double> tab_lnk_, tab_dtot_, tab_dc_, tab_db_, tab_ttot_, tab_tc_, tab_tb_;
    gsl_interp_accel *gsl_ia_dtot_, *gsl_ia_dc_, *gsl_ia_db_, *gsl_ia_ttot_, *gsl_ia_tc_, *gsl_ia_tb_;
    gsl_spline *gsl_sp_dtot_, *gsl_sp_dc_, *gsl_sp_db_, *gsl_sp_ttot_, *gsl_sp_tc_, *gsl_sp_tb_;
    double Omega_m_, Omega_b_, zstart_, ztarget_, kmax_, kmin_, h_;

    void ClassEngine_get_data( void ){
        std::vector<double> d_g, d_ur, t_g, t_ur, phi, psi;

        csoca::ilog << "Computing transfer function via ClassEngine..." << std::flush;
        double wtime = get_wtime();
        
        ClassParams pars;

        double target_redshift = 0.0;

        pars.add("z_pk",target_redshift);
        pars.add("P_k_max_h/Mpc", kmax_);
        pars.add("h",h_);
        pars.add("Omega_b",Omega_b_);
        pars.add("Omega_ur",0.0);
        pars.add("Omega_cdm",Omega_m_-Omega_b_);
        pars.add("Omega_Lambda",1.0-Omega_m_);
        pars.add("A_s",2.42e-9);
        pars.add("n_s",.96);
        pars.add("output","dTk,vTk");
        pars.add("YHe",0.25);

        std::unique_ptr<ClassEngine> CE = std::make_unique<ClassEngine>(pars, false);

        CE->getTk(target_redshift, tab_lnk_, d_g, tab_db_, tab_dc_, d_ur, tab_dtot_,
                phi, psi, t_g, tab_tb_, t_ur, tab_ttot_);
        tab_tc_ = tab_ttot_;
        #warning need to fix CDM velocities

        wtime = get_wtime() - wtime;
        csoca::ilog << "   took " << wtime << " s / " << tab_lnk_.size() << " modes."  << std::endl;
    }

public:
  explicit transfer_CLASS_plugin( ConfigFile &cf)
  : TransferFunction_plugin(cf)
  {
    h_       = cf.GetValue<double>("cosmology","H0") / 100.0; 
    Omega_m_ = cf.GetValue<double>("cosmology","Omega_m"); 
    Omega_b_ = cf.GetValue<double>("cosmology","Omega_b");
    zstart_  = cf.GetValue<double>("setup","zstart");
    ztarget_ = cf.GetValueSafe<double>("cosmology","ztarget",0.0);
    kmax_    = 1000.0;

    this->ClassEngine_get_data();
    
    gsl_ia_dtot_ = gsl_interp_accel_alloc();
    gsl_ia_dc_   = gsl_interp_accel_alloc();
    gsl_ia_db_   = gsl_interp_accel_alloc();
    gsl_ia_ttot_ = gsl_interp_accel_alloc();
    gsl_ia_tc_   = gsl_interp_accel_alloc();
    gsl_ia_tb_   = gsl_interp_accel_alloc();

    gsl_sp_dtot_ = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_dc_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_db_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_ttot_ = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_tc_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_tb_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());

    gsl_spline_init(gsl_sp_dtot_, &tab_lnk_[0], &tab_dtot_[0], tab_lnk_.size());
    gsl_spline_init(gsl_sp_dc_,   &tab_lnk_[0], &tab_dc_[0],   tab_lnk_.size());
    gsl_spline_init(gsl_sp_db_,   &tab_lnk_[0], &tab_db_[0],   tab_lnk_.size());
    gsl_spline_init(gsl_sp_ttot_, &tab_lnk_[0], &tab_ttot_[0], tab_lnk_.size());
    gsl_spline_init(gsl_sp_tc_,   &tab_lnk_[0], &tab_tc_[0],   tab_lnk_.size());
    gsl_spline_init(gsl_sp_tb_,   &tab_lnk_[0], &tab_tb_[0],   tab_lnk_.size());

    kmin_ = std::exp(tab_lnk_[0]);
    tf_distinct_ = true; // [150612SH: different density between CDM v.s. Baryon]
    tf_withvel_  = true; // [150612SH: using velocity transfer function]
  }
    
  ~transfer_CLASS_plugin(){
    gsl_spline_free(gsl_sp_dtot_);
    gsl_spline_free(gsl_sp_dc_);
    gsl_spline_free(gsl_sp_db_);
    gsl_spline_free(gsl_sp_ttot_);
    gsl_spline_free(gsl_sp_tc_);
    gsl_spline_free(gsl_sp_tb_);

    gsl_interp_accel_free(gsl_ia_dtot_);
    gsl_interp_accel_free(gsl_ia_dc_);
    gsl_interp_accel_free(gsl_ia_db_);
    gsl_interp_accel_free(gsl_ia_ttot_);
    gsl_interp_accel_free(gsl_ia_tc_);
    gsl_interp_accel_free(gsl_ia_tb_);
  }

  inline double compute(double k, tf_type type) const {
      gsl_spline *splineT = nullptr;
      gsl_interp_accel *accT = nullptr;
      switch(type){
          case total:   splineT = gsl_sp_dtot_; accT = gsl_ia_dtot_; break;
          case cdm:     splineT = gsl_sp_dc_;   accT = gsl_ia_dc_;   break;
          case baryon:  splineT = gsl_sp_db_;   accT = gsl_ia_db_;   break;
          case vtotal:  splineT = gsl_sp_ttot_; accT = gsl_ia_ttot_; break;
          case vcdm:    splineT = gsl_sp_tc_;   accT = gsl_ia_tc_;   break;
          case vbaryon: splineT = gsl_sp_tb_;   accT = gsl_ia_tb_;   break;
          default:
            throw std::runtime_error("Invalid type requested in transfer function evaluation");
      }

      double d = (k<=kmin_)? gsl_spline_eval(splineT, std::log(kmin_), accT) 
        : gsl_spline_eval(splineT, std::log(k), accT);
      return -d/(k*k);
  }

  inline double get_kmin(void) const { return std::exp(tab_lnk_[0]); }
  inline double get_kmax(void) const { return std::exp(tab_lnk_[tab_lnk_.size()-1]); }
};

namespace {
TransferFunction_plugin_creator_concrete<transfer_CLASS_plugin> creator("CLASS");
}
