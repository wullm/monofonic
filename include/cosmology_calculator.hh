#pragma once

#include <array>
#include <vec.hh>

#include <cosmology_parameters.hh>
#include <physical_constants.hh>
#include <transfer_function_plugin.hh>
#include <ode_integrate.hh>
#include <logger.hh>

#include <interpolate.hh>

#include <gsl/gsl_integration.h>
// #include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

namespace cosmology
{

/*!
 * @class cosmology::calculator
 * @brief provides functions to compute cosmological quantities
 *
 * This class provides member functions to compute cosmological quantities
 * related to the Friedmann equations and linear perturbation theory
 */
class calculator
{
public:
    //! data structure to store cosmological parameters
    cosmology::parameters cosmo_param_;

    //! pointer to an instance of a transfer function plugin
    std::unique_ptr<TransferFunction_plugin> transfer_function_;

private:
    static constexpr double REL_PRECISION = 1e-10;
    interpolated_function_1d<true,true,false> D_of_a_, f_of_a_, a_of_D_;
    double Dnow_, Dplus_start_, astart_;

    real_t integrate(double (*func)(double x, void *params), double a, double b, void *params) const
    {
        gsl_function F;
        F.function = func;
        F.params = params;

        double result;
        double error;

        gsl_set_error_handler_off();
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(100000);
        gsl_integration_qag(&F, a, b, 0, REL_PRECISION, 100000, 6, w, &result, &error);

        gsl_integration_workspace_free(w);

        gsl_set_error_handler(NULL);

        if (error / result > REL_PRECISION)
            music::wlog << "no convergence in function 'integrate', rel. error=" << error / result << std::endl;

        return (real_t)result;
    }

    void compute_growth( std::vector<double>& tab_a, std::vector<double>& tab_D, std::vector<double>& tab_f )
    {
        using v_t = vec_t<3, double>;

        // set ICs
        const double a0 = 1e-10;
        const double D0 = a0;
        const double Dprime0 = 2.0 * D0 * H_of_a(a0) / std::pow(phys_const::c_SI, 2);
        const double t0 = 1.0 / (a0 * H_of_a(a0));

        v_t y0({a0, D0, Dprime0});

        // set up integration
        double dt = 1e-9;
        double dtdid, dtnext;
        const double amax = 2.0;

        v_t yy(y0);
        double t = t0;
        const double eps = 1e-10;

        while (yy[0] < amax)
        {
            // RHS of ODEs
            auto rhs = [&](double t, v_t y) -> v_t {
                auto a = y[0];
                auto D = y[1];
                auto Dprime = y[2];
                v_t dy;
                // da/dtau = a^2 H(a)
                dy[0] = a * a * H_of_a(a);
                // d D/dtau
                dy[1] = Dprime;
                // d^2 D / dtau^2
                dy[2] = -a * H_of_a(a) * Dprime + 3.0 / 2.0 * cosmo_param_.Omega_m * std::pow(cosmo_param_.H0, 2) * D / a;
                return dy;
            };

            // scale by predicted value to get approx. constant fractional errors
            v_t yyscale = yy.abs() + dt * rhs(t, yy).abs();
            
            // call integrator
            ode_integrate::rk_step_qs(dt, t, yy, yyscale, rhs, eps, dtdid, dtnext);

            tab_a.push_back(yy[0]);
            tab_D.push_back(yy[1]);
            tab_f.push_back(yy[2]);

            dt = dtnext;
        }

        // compute f, before we stored here D'
        for (size_t i = 0; i < tab_a.size(); ++i)
        {
            tab_f[i] = tab_f[i] / (tab_a[i] * H_of_a(tab_a[i]) * tab_D[i]);
            tab_D[i] = tab_D[i];
            tab_a[i] = tab_a[i];
        }
    }

public:
    calculator() = delete;
    calculator(const calculator& c) = delete;
    //! constructor for a cosmology calculator object
    /*!
	 * @param acosmo a cosmological parameters structure
	 * @param pTransferFunction pointer to an instance of a transfer function object
	 */

    explicit calculator(config_file &cf)
        : cosmo_param_(cf), astart_( 1.0/(1.0+cf.get_value<double>("setup","zstart")) )
    {
        // pre-compute growth factors and store for interpolation
        std::vector<double> tab_a, tab_D, tab_f;
        this->compute_growth(tab_a, tab_D, tab_f);
        D_of_a_.set_data(tab_a,tab_D);
        f_of_a_.set_data(tab_a,tab_f);
        a_of_D_.set_data(tab_D,tab_a);
        Dnow_ = D_of_a_(1.0);

        Dplus_start_ = D_of_a_( astart_ ) / Dnow_;

        // set up transfer functions and compute normalisation
        transfer_function_ = std::move(select_TransferFunction_plugin(cf));
        transfer_function_->intialise();
        if( !transfer_function_->tf_isnormalised_ )
            cosmo_param_.pnorm = this->compute_pnorm_from_sigma8();
        else{
            cosmo_param_.pnorm = 1.0;
            auto sigma8 = this->compute_sigma8();
            music::ilog << "Measured sigma_8 for given PS normalisation is " <<  sigma8 << std::endl;
        }
        cosmo_param_.sqrtpnorm = std::sqrt(cosmo_param_.pnorm);

        music::ilog << std::setw(32) << std::left << "TF supports distinct CDM+baryons"
                    << " : " << (transfer_function_->tf_is_distinct() ? "yes" : "no") << std::endl;
        music::ilog << std::setw(32) << std::left << "TF maximum wave number"
                    << " : " << transfer_function_->get_kmax() << " h/Mpc" << std::endl;

        // music::ilog << "D+(MUSIC) = " << this->get_growth_factor( 1.0/(1.0+cf.get_value<double>("setup","zstart")) ) << std::endl;
        // music::ilog << "pnrom     = " << cosmo_param_.pnorm << std::endl;
    }

    ~calculator()
    {
    }

    //! Write out a correctly scaled power spectrum at time a
    void write_powerspectrum(real_t a, std::string fname) const
    {
        // const real_t Dplus0 = this->get_growth_factor(a);

        if (CONFIG::MPI_task_rank == 0)
        {
            double kmin = std::max(1e-4, transfer_function_->get_kmin());

            // write power spectrum to a file
            std::ofstream ofs(fname.c_str());
            std::stringstream ss;
            ss << " ,ap=" << a << "";
            ofs << "# " << std::setw(18) << "k [h/Mpc]"
                << std::setw(20) << ("P_dtot(k,a=ap)")
                << std::setw(20) << ("P_dcdm(k,a=ap)")
                << std::setw(20) << ("P_dbar(k,a=ap)")
                << std::setw(20) << ("P_tcdm(k,a=ap)")
                << std::setw(20) << ("P_tbar(k,a=ap)")
                << std::setw(20) << ("P_dtot(k,a=1)")
                << std::setw(20) << ("P_dcdm(k,a=1)")
                << std::setw(20) << ("P_dbar(k,a=1)")
                << std::setw(20) << ("P_tcdm(k,a=1)")
                << std::setw(20) << ("P_tbar(k,a=1)")
                << std::setw(20) << ("P_dtot(K,a=1)")
                << std::endl;
            for (double k = kmin; k < transfer_function_->get_kmax(); k *= 1.05)
            {
                ofs << std::setw(20) << std::setprecision(10) << k
                    << std::setw(20) << std::setprecision(10) << std::pow(this->get_amplitude(k, total)*Dplus_start_, 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->get_amplitude(k, cdm)*Dplus_start_, 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->get_amplitude(k, baryon)*Dplus_start_, 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->get_amplitude(k, vcdm)*Dplus_start_, 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->get_amplitude(k, vbaryon)*Dplus_start_, 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->get_amplitude(k, total0), 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->get_amplitude(k, cdm0), 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->get_amplitude(k, baryon0), 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->get_amplitude(k, vcdm0), 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->get_amplitude(k, vbaryon0), 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->get_amplitude(k, vtotal), 2.0)
                    << std::endl;
                    #warning Check whether output is at redshift that is indicated!
            }
        }
        music::ilog << "Wrote power spectrum at a=" << a << " to file \'" << fname << "\'" << std::endl;
    }

    const cosmology::parameters &get_parameters(void) const noexcept
    {
        return cosmo_param_;
    }

    //! return the value of the Hubble function H(a) = dloga/dt 
    inline double H_of_a(double a) const noexcept
    {
        double HH2 = 0.0;
        HH2 += cosmo_param_.Omega_r / (a * a * a * a);
        HH2 += cosmo_param_.Omega_m / (a * a * a);
        HH2 += cosmo_param_.Omega_k / (a * a);
        HH2 += cosmo_param_.Omega_DE * std::pow(a, -3. * (1. + cosmo_param_.w_0 + cosmo_param_.w_a)) * exp(-3. * (1.0 - a) * cosmo_param_.w_a);
        return cosmo_param_.H0 * std::sqrt(HH2);
    }

    //! Computes the linear theory growth factor D+, normalised to D+(a=1)=1
    real_t get_growth_factor(real_t a) const noexcept
    {
        return D_of_a_(a) / Dnow_;
    }

    //! Computes the inverse of get_growth_factor
    real_t get_a( real_t Dplus ) const noexcept
    {
        return a_of_D_( Dplus * Dnow_ );
    }

    //! Computes the linear theory growth rate f
    /*! Function computes (by interpolating on precalculated table)
     *   f = dlog D+ / dlog a
     */
    real_t get_f(real_t a) const noexcept
    {
        return f_of_a_(a);
    }

    //! Compute the factor relating particle displacement and velocity
    /*! Function computes
     *  vfac = a * (H(a)/h) * dlogD+ / dlog a 
     */
    real_t get_vfact(real_t a) const noexcept
    {
        return f_of_a_(a) * a * H_of_a(a) / cosmo_param_.h;
    }

    //! Integrand for the sigma_8 normalization of the power spectrum
    /*! Returns the value of the primordial power spectrum multiplied with 
     the transfer function and the window function of 8 Mpc/h at wave number k */
    static double dSigma8(double k, void *pParams)
    {
        if (k <= 0.0)
            return 0.0f;

        cosmology::calculator *pcc = reinterpret_cast<cosmology::calculator *>(pParams);

        double x = k * 8.0;
        double w = 3.0 * (sin(x) - x * cos(x)) / (x * x * x);
        static double nspect = (double)pcc->cosmo_param_.nspect;
        double tf = pcc->transfer_function_->compute(k, total);

        //... no growth factor since we compute at z=0 and normalize so that D+(z=0)=1
        return k * k * w * w * pow((double)k, (double)nspect) * tf * tf;
    }

    //! Integrand for the sigma_8 normalization of the power spectrum
    /*! Returns the value of the primordial power spectrum multiplied with 
	 the transfer function and the window function of 8 Mpc/h at wave number k */
    static double dSigma8_0(double k, void *pParams)
    {
        if (k <= 0.0)
            return 0.0f;

        cosmology::calculator *pcc = reinterpret_cast<cosmology::calculator *>(pParams);

        double x = k * 8.0;
        double w = 3.0 * (sin(x) - x * cos(x)) / (x * x * x);
        static double nspect = (double)pcc->cosmo_param_.nspect;
        double tf = pcc->transfer_function_->compute(k, total0);

        //... no growth factor since we compute at z=0 and normalize so that D+(z=0)=1
        return k * k * w * w * pow((double)k, (double)nspect) * tf * tf;
    }

    //! Computes the amplitude of a mode from the power spectrum
    /*! Function evaluates the supplied transfer function ptransfer_fun_
	 * and returns the amplitude of fluctuations at wave number k at z=0
	 * @param k wave number at which to evaluate
	 */
    inline real_t get_amplitude(real_t k, tf_type type) const
    {
        // if the transfer function doesn't need backscaling, then divide out growth factor
        real_t f = transfer_function_->tf_isnormalised_? 1.0/Dplus_start_ : 1.0;
        return f * std::pow(k, 0.5 * cosmo_param_.nspect) * transfer_function_->compute(k, type) * cosmo_param_.sqrtpnorm;
    }

    //! Computes the normalization for the power spectrum
    /*!
	 * integrates the power spectrum to fix the normalization to that given
	 * by the sigma_8 parameter
	 */
    real_t compute_sigma8(void)
    {
        real_t sigma0, kmin, kmax;
        kmax = transfer_function_->get_kmax();
        kmin = transfer_function_->get_kmin();

        if (!transfer_function_->tf_has_total0())
            sigma0 = 4.0 * M_PI * integrate(&dSigma8, (double)kmin, (double)kmax, this);
        else{
            sigma0 = 4.0 * M_PI * integrate(&dSigma8_0, (double)kmin, (double)kmax, this);
        }

        return std::sqrt(sigma0);
    }

    //! Computes the normalization for the power spectrum
    /*!
	 * integrates the power spectrum to fix the normalization to that given
	 * by the sigma_8 parameter
	 */
    real_t compute_pnorm_from_sigma8(void)
    {
        auto measured_sigma8 = this->compute_sigma8();
        return cosmo_param_.sigma8 * cosmo_param_.sigma8 / (measured_sigma8  * measured_sigma8);
    }
};

//! compute the jeans sound speed
/*! given a density in g/cm^-3 and a mass in g it gives back the sound
 *  speed in cm/s for which the input mass is equal to the jeans mass
 *  @param rho density 
 *  @param mass mass scale
 *  @returns jeans sound speed
 */
inline double jeans_sound_speed(double rho, double mass)
{
    const double G = 6.67e-8;
    return pow(6.0 * mass / M_PI * sqrt(rho) * pow(G, 1.5), 1.0 / 3.0);
}

} // namespace cosmology