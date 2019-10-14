#pragma once

namespace op{
inline auto assign_to = [](auto &g){return [&](auto i, auto v){ g[i] = v; };};
inline auto add_to = [](auto &g){return [&](auto i, auto v){ g[i] += v; };};
inline auto add_twice_to = [](auto &g){return [&](auto i, auto v){ g[i] += 2*v; };};
inline auto subtract_from = [](auto &g){return [&](auto i, auto v){ g[i] -= v; };};
inline auto subtract_twice_from = [](auto &g){return [&](auto i, auto v){ g[i] -= 2*v; };};
}
