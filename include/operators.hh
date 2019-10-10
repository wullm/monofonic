#pragma once

namespace op{
auto assign_to = [](auto &g){return [&](auto i, auto v){ g[i] = v; };};
auto add_to = [](auto &g){return [&](auto i, auto v){ g[i] += v; };};
auto add_twice_to = [](auto &g){return [&](auto i, auto v){ g[i] += 2*v; };};
auto subtract_from = [](auto &g){return [&](auto i, auto v){ g[i] -= v; };};
auto subtract_twice_from = [](auto &g){return [&](auto i, auto v){ g[i] -= 2*v; };};
}