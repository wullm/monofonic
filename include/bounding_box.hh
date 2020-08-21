// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2020 by Oliver Hahn
// 
// monofonIC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// monofonIC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <math/vec3.hh>

template <typename T>
struct bounding_box
{
    vec3_t<T> x1_, x2_;

    bounding_box(void)
    { }

    bounding_box( const vec3_t<T>& x1, const vec3_t<T>& x2)
    : x1_(x1), x2_(x2)
    { }

    bounding_box(const bounding_box &a)
    : x1_(a.x1_), x2_(a.x2_)
    { }

    bounding_box &operator/=(const bounding_box<T> &b)
    {
        for (int i = 0; i < 3; ++i)
        {
            x1_[i] = std::max<T>(x1_[i], b.x1_[i]);
            x2_[i] = std::min<T>(x2_[i], b.x2_[i]);
        }
        return *this;
    }
};

