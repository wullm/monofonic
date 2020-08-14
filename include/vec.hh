#pragma once
/*******************************************************************************\
 vec.hh - This file is part of MUSIC2 -
 a code to generate initial conditions for cosmological simulations 
 
 CHANGELOG (only majors, for details see repo):
    06/2019 - Oliver Hahn - first implementation
\*******************************************************************************/

#include <array>

//! implements general N-dim vectors of arbitrary primtive type with some arithmetic ops
template <int N, typename T = double>
struct vec_t
{
  std::array<T, N> data_;

  vec_t() {}

  vec_t(const vec_t<N, T> &v)
      : data_(v.data_) {}

  vec_t(vec_t<N, T> &&v)
      : data_(std::move(v.data_)) {}

  template <typename... E>
  vec_t(E... e)
      : data_{{std::forward<E>(e)...}}
  {
    static_assert(sizeof...(E) == N, "Brace-enclosed initialiser list doesn't match vec_t length!");
  }

  //! bracket index access to vector components
  T &operator[](size_t i) noexcept { return data_[i]; }

  //! const bracket index access to vector components
  const T &operator[](size_t i) const noexcept { return data_[i]; }

  // assignment operator
  vec_t<N, T> &operator=(const vec_t<N, T> &v) noexcept
  {
    data_ = v.data_;
    return *this;
  }

  //! implementation of summation of vec_t
  vec_t<N, T> operator+(const vec_t<N, T> &v) const noexcept
  {
    vec_t<N, T> res;
    for (int i = 0; i < N; ++i)
      res[i] = data_[i] + v[i];
    return res;
  }

  //! implementation of difference of vec_t
  vec_t<N, T> operator-(const vec_t<N, T> &v) const noexcept
  {
    vec_t<N, T> res;
    for (int i = 0; i < N; ++i)
      res[i] = data_[i] - v[i];
    return res;
  }

  //! implementation of unary negative
  vec_t<N, T> operator-() const noexcept
  {
    vec_t<N, T> res;
    for (int i = 0; i < N; ++i)
      res[i] = -data_[i];
    return res;
  }

  //! implementation of scalar multiplication
  template <typename T2>
  vec_t<N, T> operator*(T2 s) const noexcept
  {
    vec_t<N, T> res;
    for (int i = 0; i < N; ++i)
      res[i] = data_[i] * s;
    return res;
  }

  //! implementation of scalar division
  vec_t<N, T> operator/(T s) const noexcept
  {
    vec_t<N, T> res;
    for (int i = 0; i < N; ++i)
      res[i] = data_[i] / s;
    return res;
  }

  //! takes the absolute value of each element
  vec_t<N, T> abs(void) const noexcept
  {
    vec_t<N, T> res;
    for (int i = 0; i < N; ++i)
      res[i] = std::fabs(data_[i]);
    return res;
  }

  //! implementation of implicit summation of vec_t
  vec_t<N, T> &operator+=(const vec_t<N, T> &v) noexcept
  {
    for (int i = 0; i < N; ++i)
      data_[i] += v[i];
    return *this;
  }

  //! implementation of implicit subtraction of vec_t
  vec_t<N, T> &operator-=(const vec_t<N, T> &v) noexcept
  {
    for (int i = 0; i < N; ++i)
      data_[i] -= v[i];
    return *this;
  }

  //! implementation of implicit scalar multiplication of vec_t
  vec_t<N, T> &operator*=(T s) noexcept
  {
    for (int i = 0; i < N; ++i)
      data_[i] *= s;
    return *this;
  }

  //! implementation of implicit scalar division of vec_t
  vec_t<N, T> &operator/=(T s) noexcept
  {
    for (int i = 0; i < N; ++i)
      data_[i] /= s;
    return *this;
  }

  size_t size(void) const noexcept { return N; }
};

//! multiplication with scalar
template <typename T2, int N, typename T = double>
inline vec_t<N, T> operator*(T2 s, const vec_t<N, T> &v)
{
  vec_t<N, T> res;
  for (int i = 0; i < N; ++i)
    res[i] = v[i] * s;
  return res;
}
