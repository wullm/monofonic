#pragma once
/*
 
 operators.hh - This file is part of MUSIC2 -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2019  Oliver Hahn
 
*/
#include <general.hh>

namespace op{

//!== list of primitive operators to work on fields ==!//

template< typename field>
inline auto assign_to( field& g ){return [&g](auto i, auto v){ g[i] = v; };}

template< typename field, typename val >
inline auto multiply_add_to( field& g, val x ){return [&g,x](auto i, auto v){ g[i] += v*x; };}

template< typename field>
inline auto add_to( field& g ){return [&g](auto i, auto v){ g[i] += v; };}

template< typename field>
inline auto subtract_from( field& g ){return [&g](auto i, auto v){ g[i] -= v; };}

//! vanilla standard gradient
class fourier_gradient{
private:
    real_t boxlen_, k0_;
    size_t n_, nhalf_;
public:
    explicit fourier_gradient( const config_file& the_config )
    : boxlen_( the_config.get_value<double>("setup", "BoxLength") ), 
      k0_(2.0*M_PI/boxlen_),
      n_( the_config.get_value<size_t>("setup","GridRes") ),
      nhalf_( n_/2 )
    {}

    inline ccomplex_t gradient( const int idim, std::array<size_t,3> ijk ) const
    {
        real_t rgrad = 
            (ijk[idim]!=nhalf_)? (real_t(ijk[idim]) - real_t(ijk[idim] > nhalf_) * n_) : 0.0; 
        return ccomplex_t(0.0,rgrad * k0_);
    }

    inline real_t vfac_corr( std::array<size_t,3> ijk ) const
    {
        return 1.0;
    }
};
}
