/*******************************************************************************
 * This file is part of 3FA.
 * Copyright (c) 2021 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>

#include "fluid_equations.h"
#include "strooklat.h"

struct ode_params {
    struct strooklat *spline;
    struct cosmology_tables *tab;
    double k;
    double f_b;
    double c_s;
};

int func (double loga, const double y[], double f[], void *params) {
    struct ode_params *p = (struct ode_params *) params;
    struct strooklat *spline = p->spline;
    struct cosmology_tables *tab = p->tab;

    double a = exp(loga);
    double A = strooklat_interp(spline, tab->Avec, a);
    double B = strooklat_interp(spline, tab->Bvec, a);
    double H = strooklat_interp(spline, tab->Hvec, a);
    double f_nu_nr = strooklat_interp(spline, tab->f_nu_nr, a);

    double c_s = p->c_s / a;
    double k = p->k;
    double k_fs2 = -B * H * H / (c_s * c_s) * (a * a);
    double f_b = p->f_b;
    double D_cb = (1.0 - f_b) * y[0] + f_b * y[2];

    f[0] = -y[1];
    f[1] = A * y[1] + B * ((1.0 - f_nu_nr) * D_cb + f_nu_nr * y[4]);
    f[2] = -y[3];
    f[3] =  A * y[3] + B * ((1.0 - f_nu_nr) * D_cb + f_nu_nr * y[4]);
    f[4] = -y[5];
    f[5] = A * y[5] + B * ((1.0 - f_nu_nr) * D_cb + (f_nu_nr - (k*k)/k_fs2)*y[4]);

    return GSL_SUCCESS;
}

void integrate_fluid_equations(struct model *m, struct units *us,
                               struct cosmology_tables *tab,
                               struct growth_factors *gfac,
                               double a_start, double a_final) {

    /* We will differentiate the density perturbations at a_start */
    double log_a_start = log(a_start);
    double delta_log_a = 0.002;

    /* Nearby scale factors */
    double a_mm = exp(log_a_start - 2.0 * delta_log_a);
    double a_m = exp(log_a_start - 1.0 * delta_log_a);
    double a_p = exp(log_a_start + 1.0 * delta_log_a);
    double a_pp = exp(log_a_start + 2.0 * delta_log_a);

    /* Create a scale factor spline for the cosmological tables */
    struct strooklat spline_tab = {tab->avec, tab->size};
    init_strooklat_spline(&spline_tab, 100);

    /* Compute the Hubble ratio */
    double H_start = strooklat_interp(&spline_tab, tab->Hvec, a_start);
    double H_final = strooklat_interp(&spline_tab, tab->Hvec, a_final);

    /* Prepare the parameters for the fluid ODEs */
    struct ode_params odep;
    odep.spline = &spline_tab;
    odep.tab = tab;
    odep.f_b = m->Omega_b / (m->Omega_c + m->Omega_b);
    odep.c_s = 134.423 / m->M_nu[0] * KM_METRES / us->UnitLengthMetres * us->UnitTimeSeconds;

    /* The wavenumber of interest */
    double k = gfac->k;
    odep.k = k;

    /* Find the values at the starting redshift and normalize */
    double Dc = gfac->Dc;
    double Db = gfac->Db;
    double Dn = gfac->Dn;

    /* Initial conditions */
    double beta_c = Dc / Dc;
    double beta_b = Db / Dc;
    double beta_n = Dn / Dc;

    /* Growth rates at a_start */
    double gc = gfac->gc;
    double gb = gfac->gb;
    double gn = gfac->gn;

    /* Prepare the initial conditions */
    double y[6] = {beta_c, -gc * beta_c, beta_b, -gb * beta_b, beta_n, -gn * beta_n};
    double loga = log(a_start);
    double loga_final = log(a_final);

    /* Integrate */
    double tol = 1e-12;
    double hstart = 1e-12;
    gsl_odeiv2_system sys = {func, NULL, 6, &odep};
    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, hstart, tol, tol);
    gsl_odeiv2_driver_apply(d, &loga, loga_final, y);
    gsl_odeiv2_driver_free(d);

    /* Extract the result */
    double Dc_final = y[0];
    double Db_final = y[2];
    double Dn_final = y[4];
    double gc_final = -y[1]/y[0];
    double gb_final = -y[3]/y[2];
    double gn_final = -y[5]/y[4];

    /* Store the results (relative density and velocity transfer functions)*/
    gfac->Dc = beta_c / Dc_final;
    gfac->Db = beta_b / Db_final;
    gfac->Dn = beta_n / Dn_final;

    /* Free the perturbation splines */
    free_strooklat_spline(&spline_tab);
}