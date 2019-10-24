#pragma once

namespace op{

template< typename grid>
inline auto assign_to( grid& g ){return [&](auto i, auto v){ g[i] = v; };}

template< typename grid>
inline auto add_to( grid& g ){return [&](auto i, auto v){ g[i] += v; };}

template< typename grid>
inline auto add_twice_to( grid& g ){return [&](auto i, auto v){ g[i] += 2*v; };}

template< typename grid>
inline auto subtract_from( grid& g ){return [&](auto i, auto v){ g[i] -= v; };}

template< typename grid>
inline auto subtract_twice_from( grid& g ){return [&](auto i, auto v){ g[i] -= 2*v; };}

// above template functions can be written as C++17 inline lambdas... but we're using C++14...
// inline auto assign_to = [](auto &g){return [&](auto i, auto v){ g[i] = v; };};
// inline auto add_to = [](auto &g){return [&](auto i, auto v){ g[i] += v; };};
// inline auto add_twice_to = [](auto &g){return [&](auto i, auto v){ g[i] += 2*v; };};
// inline auto subtract_from = [](auto &g){return [&](auto i, auto v){ g[i] -= v; };};
// inline auto subtract_twice_from = [](auto &g){return [&](auto i, auto v){ g[i] -= 2*v; };};
}
