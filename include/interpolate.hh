#pragma once

#include <vector>
#include <cassert>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

template <bool logx, bool logy, bool periodic>
class interpolated_function_1d
{

private:
  bool isinit_;
  std::vector<double> data_x_, data_y_;
  gsl_interp_accel *gsl_ia_;
  gsl_spline *gsl_sp_;

  void deallocate()
  {
    gsl_spline_free(gsl_sp_);
    gsl_interp_accel_free(gsl_ia_);
  }

public:
  interpolated_function_1d(const interpolated_function_1d &) = delete;

  interpolated_function_1d() : isinit_(false){}

  interpolated_function_1d(const std::vector<double> &data_x, const std::vector<double> &data_y)
  : isinit_(false)
  {
    this->set_data( data_x, data_y );
  }

  ~interpolated_function_1d()
  {
    if (isinit_) this->deallocate();
  }

  void set_data(const std::vector<double> &data_x, const std::vector<double> &data_y)
  {
    assert(data_x_.size() == data_y_.size());
    assert(!(logx & periodic));

    data_x_ = data_x;
    data_y_ = data_y;

    if (logx) for (auto &d : data_x_) d = std::log(d);
    if (logy) for (auto &d : data_y_) d = std::log(d);

    if (isinit_) this->deallocate();

    gsl_ia_ = gsl_interp_accel_alloc();
    gsl_sp_ = gsl_spline_alloc(periodic ? gsl_interp_cspline_periodic : gsl_interp_cspline, data_x_.size());
    gsl_spline_init(gsl_sp_, &data_x_[0], &data_y_[0], data_x_.size());

    isinit_ = true;
  }

  double operator()(double x) const noexcept
  {
    double xa = logx ? std::log(x) : x;
    double y(gsl_spline_eval(gsl_sp_, xa, gsl_ia_));
    return logy ? std::exp(y) : y;
  }
};