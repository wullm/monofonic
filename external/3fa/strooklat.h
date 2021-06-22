/*******************************************************************************
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

/*
 *  @file strooklat_inline.h
 *  @brief Inline methods for fast linear interpolation.
 */

#ifndef STROOKLAT_SPLINE_H
#define STROOKLAT_SPLINE_H

/* Standard libraries */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Spline struct with lookup table */
struct strooklat {
    /* The x values of the data points to be interpolated */
    const double *x;
    /* The number of data points */
    const int size;
    /* The last index returned */
    int last_index;

    struct lookup {
        /* A lookup table for faster searches */
        int *lookup_table;
        /* Size of the lookup table */
        int lookup_table_size;
    } lookup;
};

/* Identity map if ascending, reverses the order if descending */
static inline int sorted_id(int i, int size, int ascend) {
    return ascend ? i : size - 1 - i;
}

/**
 * @brief Initialize the lookup table for the given strooklat spline struct.
 *
 * @param spline The spline to be initialized
 * @param lookup_size Size of the lookup table
 */
static inline int init_strooklat_spline(struct strooklat *spline,
                                        int lookup_size) {
    /* Size of the x-values array */
    int size = spline->size;

    /* Are the x-values ascending or descending? */
    int ascend = (spline->x[0] < spline->x[size - 1]);

    /* Check that the x-values are sorted */
    for (int i = 0; i < size - 1; i++) {
        if ((ascend && spline->x[i] > spline->x[i + 1]) ||
                (!ascend && spline->x[i] < spline->x[i + 1])) {
            printf("Error: x values are not sorted.\n");
            return 1;
        }
    }

    /* Set the cached index to a definite value */
    spline->last_index = 0;

    /* Allocate the lookup table */
    spline->lookup.lookup_table = malloc(lookup_size * sizeof(int));
    spline->lookup.lookup_table_size = lookup_size;

    if (spline->lookup.lookup_table == NULL) {
        printf("Error: could not allocate lookup table.\n");
        return 1;
    }

    /* Retrieve the minimum and maximum elements of the x array */
    double x_min = spline->x[sorted_id(0, size, ascend)];
    double x_max = spline->x[sorted_id(0, size, !ascend)];

    /* Create the lookup table */
    for (int i = 0; i < lookup_size; i++) {
        /* Map i in [0, lookup_size - 1] to v in [x_min, x_max] */
        double u = (double)i / lookup_size;
        double v = x_min + u * (x_max - x_min);
        int j;

        /* Find the smallest j such that x[j-1] < v and x[j] >= v */
        for (j = 0; j < size && spline->x[sorted_id(j, size, ascend)] < v; j++)
            ;

        /* Store the index */
        if (j == 0) {
            spline->lookup.lookup_table[i] = 0;
        } else {
            spline->lookup.lookup_table[i] = j - 1;
        }
    }

    return 0;
}

/**
 * @brief Clean up the spline
 *
 * @param spline The spline in question
 */
static inline int free_strooklat_spline(struct strooklat *spline) {
    free(spline->lookup.lookup_table);
    return 0;
}

/**
 * @brief Find an interval containing the given x-value and compute the
 * ratio u = (x - x_left) / (x_right - x_left)
 *
 * @param spline The spline in question
 * @param x The x value to be located
 * @param ind Reference to index (output)
 * @param u Reference to ratio (output)
 */
static inline int strooklat_find_x(struct strooklat *spline, double x, int *ind,
                                   double *u) {

    /* Sizes of the lookup table and full array */
    int look_size = spline->lookup.lookup_table_size;
    int size = spline->size;
    int ascend = (spline->x[0] < spline->x[size - 1]);

    /* Bounding values for the x-array */
    double x_min = spline->x[sorted_id(0, size, ascend)];
    double x_max = spline->x[sorted_id(0, size, !ascend)];

    /* Quickly return if the x value is out of bounds */
    if (x > x_max) {
        *ind = size - 2;
        *u = 1.0;
    } else if (x < x_min) {
        *ind = 0;
        *u = 0.0;
    }

    /* Quickly check the last index to see if it still works */
    if (x > spline->x[sorted_id(spline->last_index, size, ascend)] &&
            x <= spline->x[sorted_id(spline->last_index + 1, size, ascend)]) {
        *ind = spline->last_index;
    } else {
        /* Quickly find a starting index using the lookup table */
        double w = (x - x_min) / (x_max - x_min);
        int i = floor(w * look_size);
        int j = spline->lookup.lookup_table[i < look_size ? i : look_size - 1];

        /* Find the smallest j such that x[j-1] < x and x[j] >= x */
        for (j = j; j < size && spline->x[sorted_id(j, size, ascend)] < x; j++)
            ;

        /* We found the index */
        *ind = j - 1;

        /* Cache this index */
        spline->last_index = *ind;
    }

    /* Find the bounding values */
    double left = spline->x[sorted_id(*ind, size, ascend)];
    double right = spline->x[sorted_id(*ind + 1, size, ascend)];

    /* Calculate the ratio (X - X_left) / (X_right - X_left) */
    *u = (x - left) / (right - left);

    return 0;
}

/**
 * @brief Linearly interpolate the y values given the closest x-index and the
 * ratio u = (x - x_left) / (x_right - x_left)
 *
 * @param spline The spline in question
 * @param y The array of y values (should be same size as x)
 * @param ind The x-index
 * @param u The ratio u along the interval
 */
static inline double strooklat_interp_index(struct strooklat *spline, double *y,
        int ind, double u) {

    /* Retrieve the bounding values */
    int size = spline->size;
    int ascend = (spline->x[0] < spline->x[size - 1]);
    double left = y[sorted_id(ind, size, ascend)];
    double right = y[sorted_id(ind + 1, size, ascend)];

    return (1 - u) * left + u * right;
}

/**
 * @brief Linearly interpolate the y values at the given x value
 *
 * @param spline The spline in question
 * @param y The array of y values (should be same size as x)
 * @param x The x value
 */
static inline double strooklat_interp(struct strooklat *spline, double *y,
                                      double x) {

    /* Find the bounding interval */
    int ind;
    double u;
    strooklat_find_x(spline, x, &ind, &u);

    /* Interpolate the y-value */
    return strooklat_interp_index(spline, y, ind, u);
}

/**
 * @brief Bi-linearly interpolate the z values at the given x and y values
 *
 * @param spline_x The spline for the x values
 * @param spline_y The spline for the x values
 * @param z The array of z values (should be of size x * y)
 * @param x The x value
 * @param y The y value
 */
static inline double strooklat_interp_2d(struct strooklat *spline_x,
                                         struct strooklat *spline_y, double *z,
                                         double x, double y) {

    /* Find the bounding interval for x */
    int ind_x;
    double u_x;
    strooklat_find_x(spline_x, x, &ind_x, &u_x);
    
    /* Find the bounding interval for y */
    int ind_y;
    double u_y;
    strooklat_find_x(spline_y, y, &ind_y, &u_y);
    
    /* Determine the sizes and orderings of the two dimensions */
    int size_x = spline_x->size;
    int size_y = spline_y->size;
    int ascend_x = (spline_x->x[0] < spline_x->x[size_x - 1]);
    int ascend_y = (spline_y->x[0] < spline_y->x[size_y - 1]);
    
    /* Retrieve the bounding values */
    int left_x = sorted_id(ind_x, size_x, ascend_x);
    int right_x = sorted_id(ind_x + 1, size_x, ascend_x);
    int left_y = sorted_id(ind_y, size_y, ascend_y);
    int right_y = sorted_id(ind_y + 1, size_y, ascend_y);
    
    /* Retrieve the bounding values */
    double T11 = z[size_y * left_x + left_y];
    double T21 = z[size_y * left_x + right_y];
    double T12 = z[size_y * right_x + left_y];
    double T22 = z[size_y * right_x + right_y];

    return (1 - u_x) * ((1 - u_y) * T11 + u_y * T21)
               + u_x * ((1 - u_y) * T12 + u_y * T22);
}

#endif
