//  transfer_CLASS.cc - This file is part of MUSIC -
//  a code to generate multi-scale initial conditions for cosmological simulations

//  Copyright (C) 2019  Oliver Hahn

#ifdef USE_CLASS

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

class transfer_CLASS_plugin : public TransferFunction_plugin {

private:
    //... target redshift tables
    std::vector<double> tab_lnk_, tab_dtot_, tab_dc_, tab_db_, tab_ttot_, tab_tc_, tab_tb_;
    gsl_interp_accel *gsl_ia_dtot_, *gsl_ia_dc_, *gsl_ia_db_, *gsl_ia_ttot_, *gsl_ia_tc_, *gsl_ia_tb_;
    gsl_spline *gsl_sp_dtot_, *gsl_sp_dc_, *gsl_sp_db_, *gsl_sp_ttot_, *gsl_sp_tc_, *gsl_sp_tb_;

    //... starting redshift tables
    std::vector<double> tab_lnk0_, tab_dtot0_, tab_dc0_, tab_db0_, tab_ttot0_, tab_tc0_, tab_tb0_;
    gsl_interp_accel *gsl_ia_dtot0_, *gsl_ia_dc0_, *gsl_ia_db0_, *gsl_ia_ttot0_, *gsl_ia_tc0_, *gsl_ia_tb0_;
    gsl_spline *gsl_sp_dtot0_, *gsl_sp_dc0_, *gsl_sp_db0_, *gsl_sp_ttot0_, *gsl_sp_tc0_, *gsl_sp_tb0_;

    // single fluid growing/decaying mode decomposition
    gsl_interp_accel *gsl_ia_Cplus_, *gsl_ia_Cminus_;
    gsl_spline *gsl_sp_Cplus_, *gsl_sp_Cminus_;
    std::vector<double> tab_Cplus_, tab_Cminus_;

    double Omega_m_, Omega_b_, N_ur_, zstart_, ztarget_, kmax_, kmin_, h_, astart_, atarget_;

    void ClassEngine_get_data( void ){
        std::vector<double> d_ncdm, t_ncdm, phi, psi;

        csoca::ilog << "Computing TF via ClassEngine..." << std::endl << " ztarget = " << ztarget_ << ", zstart = " << zstart_ << " ..." << std::flush;
        double wtime = get_wtime();
        
        ClassParams pars;
        pars.add("extra metric transfer functions", "yes");
        pars.add("z_pk",ztarget_);
        pars.add("P_k_max_h/Mpc", kmax_);
        
        pars.add("h",h_);
        pars.add("Omega_b",Omega_b_);
        // pars.add("Omega_k",0.0);
        // pars.add("Omega_ur",0.0);
        pars.add("N_ur",N_ur_);
        pars.add("Omega_cdm",Omega_m_-Omega_b_);
        pars.add("Omega_k",0.0);
        // pars.add("Omega_Lambda",1.0-Omega_m_);
        pars.add("Omega_fld",0.0);
        pars.add("Omega_scf",0.0);

        pars.add("A_s",2.42e-9);
        pars.add("n_s",.961); // this doesn't matter for TF
        pars.add("output","dTk,vTk");
        pars.add("YHe",0.248);
        pars.add("lensing","no");
        pars.add("alpha_s",0.0);
        pars.add("P_k_ini type","analytic_Pk");
        pars.add("gauge","synchronous");

        pars.add("k_per_decade_for_pk",100);
        pars.add("k_per_decade_for_bao",100);

        pars.add("compute damping scale","yes");
        pars.add("z_reio",-1.0); // make sure reionisation is not included

        pars.add("tol_perturb_integration",1.e-8);
        pars.add("tol_background_integration",1e-9);

        // high precision options from cl_permille.pre:
        // precision file to be passed as input in order to achieve at least percent precision on scalar Cls
        pars.add("hyper_flat_approximation_nu", 7000. );
        pars.add("transfer_neglect_delta_k_S_t0", 0.17 );
        pars.add("transfer_neglect_delta_k_S_t1", 0.05 );
        pars.add("transfer_neglect_delta_k_S_t2", 0.17 );
        pars.add("transfer_neglect_delta_k_S_e", 0.13 );
        pars.add("delta_l_max", 1000 );


        std::unique_ptr<ClassEngine> CE = std::make_unique<ClassEngine>(pars, false);

        CE->getTk(zstart_, tab_lnk0_, tab_dc0_, tab_db0_, d_ncdm, tab_dtot0_,
                tab_tc0_, tab_tb0_, t_ncdm, tab_ttot0_, phi, psi );

        CE->getTk(ztarget_, tab_lnk_, tab_dc_, tab_db_, d_ncdm, tab_dtot_,
                  tab_tc_, tab_tb_, t_ncdm, tab_ttot_, phi, psi);

        wtime = get_wtime() - wtime;
        csoca::ilog << "   took " << wtime << " s / " << tab_lnk_.size() << " modes."  << std::endl;
    }

public:
  explicit transfer_CLASS_plugin( ConfigFile &cf)
  : TransferFunction_plugin(cf)
  { 
    h_       = pcf_->GetValue<double>("cosmology","H0") / 100.0; 
    Omega_m_ = pcf_->GetValue<double>("cosmology","Omega_m"); 
    Omega_b_ = pcf_->GetValue<double>("cosmology","Omega_b");
    N_ur_    = pcf_->GetValueSafe<double>("cosmology","N_ur", 3.046);
    ztarget_ = pcf_->GetValueSafe<double>("cosmology","ztarget",0.0);
    atarget_ = 1.0/(1.0+ztarget_);
    zstart_  = pcf_->GetValue<double>("setup","zstart");
    astart_  = 1.0/(1.0+zstart_);
    double lbox = pcf_->GetValue<double>("setup","BoxLength");
    int nres = pcf_->GetValue<double>("setup","GridRes");
    kmax_    = 2.0*M_PI/lbox * nres/2 * sqrt(3) * 2.0; // 120% of spatial diagonal

    this->ClassEngine_get_data();
    
    gsl_ia_dtot_ = gsl_interp_accel_alloc();  gsl_ia_dtot0_ = gsl_interp_accel_alloc();
    gsl_ia_dc_   = gsl_interp_accel_alloc();  gsl_ia_dc0_   = gsl_interp_accel_alloc();
    gsl_ia_db_   = gsl_interp_accel_alloc();  gsl_ia_db0_   = gsl_interp_accel_alloc();
    gsl_ia_ttot_ = gsl_interp_accel_alloc();  gsl_ia_ttot0_ = gsl_interp_accel_alloc();
    gsl_ia_tc_   = gsl_interp_accel_alloc();  gsl_ia_tc0_   = gsl_interp_accel_alloc();
    gsl_ia_tb_   = gsl_interp_accel_alloc();  gsl_ia_tb0_   = gsl_interp_accel_alloc();

    gsl_sp_dtot_ = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_dc_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_db_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_ttot_ = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_tc_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_tb_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());

    gsl_sp_dtot0_ = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_dc0_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_db0_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_ttot0_ = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_tc0_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_tb0_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());

    gsl_spline_init(gsl_sp_dtot_, &tab_lnk_[0], &tab_dtot_[0], tab_lnk_.size());
    gsl_spline_init(gsl_sp_dc_,   &tab_lnk_[0], &tab_dc_[0],   tab_lnk_.size());
    gsl_spline_init(gsl_sp_db_,   &tab_lnk_[0], &tab_db_[0],   tab_lnk_.size());
    gsl_spline_init(gsl_sp_ttot_, &tab_lnk_[0], &tab_ttot_[0], tab_lnk_.size());
    gsl_spline_init(gsl_sp_tc_,   &tab_lnk_[0], &tab_tc_[0],   tab_lnk_.size());
    gsl_spline_init(gsl_sp_tb_,   &tab_lnk_[0], &tab_tb_[0],   tab_lnk_.size());

    gsl_spline_init(gsl_sp_dtot0_, &tab_lnk0_[0], &tab_dtot0_[0], tab_lnk0_.size());
    gsl_spline_init(gsl_sp_dc0_,   &tab_lnk0_[0], &tab_dc0_[0],   tab_lnk0_.size());
    gsl_spline_init(gsl_sp_db0_,   &tab_lnk0_[0], &tab_db0_[0],   tab_lnk0_.size());
    gsl_spline_init(gsl_sp_ttot0_, &tab_lnk0_[0], &tab_ttot0_[0], tab_lnk0_.size());
    gsl_spline_init(gsl_sp_tc0_,   &tab_lnk0_[0], &tab_tc0_[0],   tab_lnk0_.size());
    gsl_spline_init(gsl_sp_tb0_,   &tab_lnk0_[0], &tab_tb0_[0],   tab_lnk0_.size());

    //--------------------------------------------------------------------------
    // single fluid growing/decaying mode decomposition
    //--------------------------------------------------------------------------
    gsl_ia_Cplus_  = gsl_interp_accel_alloc();
    gsl_ia_Cminus_ = gsl_interp_accel_alloc();
    
    gsl_sp_Cplus_  = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_Cminus_ = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    
    tab_Cplus_.assign(tab_lnk_.size(),0);
    tab_Cminus_.assign(tab_lnk_.size(),0);

    std::ofstream ofs("grow_decay.txt");
    
    for( size_t i=0; i<tab_lnk_.size(); ++i ){
      tab_Cplus_[i]  = (3.0/5.0 * tab_dtot_[i]/atarget_ - 2.0/5.0*tab_ttot_[i]/atarget_);
      tab_Cminus_[i] = (2.0/5.0 * std::pow(atarget_, 1.5) *  ( tab_dtot_[i] + tab_ttot_[i] ));

      ofs << std::exp(tab_lnk_[i]) << " " << tab_Cplus_[i] << " " << tab_Cminus_[i] << " " << tab_dtot_[i] << " " << tab_ttot_[i] << std::endl;
    }
    
    gsl_spline_init(gsl_sp_Cplus_,   &tab_lnk_[0], &tab_Cplus_[0],   tab_lnk_.size());
    gsl_spline_init(gsl_sp_Cminus_,  &tab_lnk_[0], &tab_Cminus_[0],  tab_lnk_.size());
    //--------------------------------------------------------------------------
    
    

    kmin_ = std::exp(tab_lnk_[0]);
  
    tf_distinct_ = true; 
    tf_withvel_  = true; 
  }
    
  ~transfer_CLASS_plugin(){
    gsl_spline_free(gsl_sp_dtot_);   gsl_spline_free(gsl_sp_dtot0_);  
    gsl_spline_free(gsl_sp_dc_);     gsl_spline_free(gsl_sp_dc0_);
    gsl_spline_free(gsl_sp_db_);     gsl_spline_free(gsl_sp_db0_);
    gsl_spline_free(gsl_sp_ttot_);   gsl_spline_free(gsl_sp_ttot0_);  
    gsl_spline_free(gsl_sp_tc_);     gsl_spline_free(gsl_sp_tc0_);
    gsl_spline_free(gsl_sp_tb_);     gsl_spline_free(gsl_sp_tb0_);

    gsl_interp_accel_free(gsl_ia_dtot_);  gsl_interp_accel_free(gsl_ia_dtot0_);  
    gsl_interp_accel_free(gsl_ia_dc_);    gsl_interp_accel_free(gsl_ia_dc0_);
    gsl_interp_accel_free(gsl_ia_db_);    gsl_interp_accel_free(gsl_ia_db0_);
    gsl_interp_accel_free(gsl_ia_ttot_);  gsl_interp_accel_free(gsl_ia_ttot0_);  
    gsl_interp_accel_free(gsl_ia_tc_);    gsl_interp_accel_free(gsl_ia_tc0_);
    gsl_interp_accel_free(gsl_ia_tb_);    gsl_interp_accel_free(gsl_ia_tb0_);
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
          
          case total0:  splineT = gsl_sp_dtot0_;accT = gsl_ia_dtot0_;break;
          case cdm0:    splineT = gsl_sp_dc0_;  accT = gsl_ia_dc0_;  break;
          case baryon0: splineT = gsl_sp_db0_;  accT = gsl_ia_db0_;  break;
          case vtotal0: splineT = gsl_sp_ttot0_;accT = gsl_ia_ttot0_;break;
          case vcdm0:   splineT = gsl_sp_tc0_;  accT = gsl_ia_tc0_;  break;
          case vbaryon0:splineT = gsl_sp_tb0_;  accT = gsl_ia_tb0_;  break;
          default:
            throw std::runtime_error("Invalid type requested in transfer function evaluation");
      }

      double d = (k<=kmin_)? gsl_spline_eval(splineT, std::log(kmin_), accT) 
        : gsl_spline_eval(splineT, std::log(k*h_), accT);
      return -d/(k*k);
  }

  inline double get_kmin(void) const { return std::exp(tab_lnk_[0])/h_; }
  inline double get_kmax(void) const { return std::exp(tab_lnk_[tab_lnk_.size()-1])/h_; }
};

namespace {
TransferFunction_plugin_creator_concrete<transfer_CLASS_plugin> creator("CLASS");
}

#endif // USE_CLASS
