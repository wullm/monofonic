#pragma once

#include <vec3.hh>

template <typename T>
struct bounding_box
{
    vec3<T> x1_, x2_;

    bounding_box(void)
    { }

    bounding_box( const vec3<T>& x1, const vec3<T>& x2)
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

