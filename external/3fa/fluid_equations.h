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

#ifndef FLUID_EQUATIONS_H
#define FLUID_EQUATIONS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "units.h"
#include "cosmology_tables.h"

struct growth_factors {
    /* Wavenumber */
    double k;
    /* Density transfer function ratios D_start(k) / D_final(k) */
    double Dc;
    double Db;
    double Dn;
    /* Growth rates at a_start */
    double gc;
    double gb;
    double gn;
};

void integrate_fluid_equations(struct model *m, struct units *us,
                               struct cosmology_tables *tab,
                               struct growth_factors *gfac,
                               double a_start, double a_final);
                               
#ifdef __cplusplus
}
#endif

#endif
