#pragma once

namespace op{

template< typename field>
inline auto assign_to( field& g ){return [&g](auto i, auto v){ g[i] = v; };}

template< typename field, typename val >
inline auto multiply_add_to( field& g, val x ){return [&g,x](auto i, auto v){ g[i] += v*x; };}

template< typename field>
inline auto add_to( field& g ){return [&g](auto i, auto v){ g[i] += v; };}

template< typename field>
inline auto add_twice_to( field& g ){return [&g](auto i, auto v){ g[i] += 2*v; };}

template< typename field>
inline auto subtract_from( field& g ){return [&g](auto i, auto v){ g[i] -= v; };}

template< typename field>
inline auto subtract_twice_from( field& g ){return [&g](auto i, auto v){ g[i] -= 2*v; };}

}
