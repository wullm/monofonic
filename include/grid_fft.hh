#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <execution>

#include <vec3.hh>
#include <general.hh>
#include <bounding_box.hh>

enum space_t
{
    kspace_id,
    rspace_id
};


template <typename data_t>
class Grid_FFT
{
protected:
#if defined(USE_MPI)
    const MPI_Datatype MPI_data_t_type = (typeid(data_t) == typeid(double)) ? MPI_DOUBLE
       : (typeid(data_t) == typeid(float)) ? MPI_FLOAT
       : (typeid(data_t) == typeid(std::complex<float>)) ? MPI_COMPLEX
       : (typeid(data_t) == typeid(std::complex<double>)) ? MPI_DOUBLE_COMPLEX : MPI_INT;
#endif
public:
    std::array<size_t, 3> n_, nhalf_;
    std::array<size_t, 4> sizes_;
    size_t npr_, npc_;
    size_t ntot_;
    std::array<real_t, 3> length_, kfac_, dx_;

    space_t space_;
    data_t *data_;
    ccomplex_t *cdata_;

    bounding_box<size_t> global_range_;

    fftw_plan_t plan_, iplan_;

    real_t fft_norm_fac_;

    ptrdiff_t local_0_start_, local_1_start_;
    ptrdiff_t local_0_size_, local_1_size_;

    Grid_FFT(const std::array<size_t, 3> &N, const std::array<real_t, 3> &L, space_t initialspace = rspace_id)
        : n_(N), length_(L), space_(initialspace), data_(nullptr), cdata_(nullptr) 
    {
        //invalidated = true;
        this->Setup();
    }

    // avoid implicit copying of data
    Grid_FFT(const Grid_FFT<data_t> &g) = delete;

    ~Grid_FFT()
    {
        if (data_ != nullptr)
        {
            fftw_free(data_);
        }
    }

    const Grid_FFT<data_t>* get_grid( size_t ilevel ) const { return this; }
    bool is_in_mask( size_t ilevel, size_t i, size_t j, size_t k ) const { return true; }
    bool is_refined( size_t ilevel, size_t i, size_t j, size_t k ) const { return false; }
    size_t levelmin() const {return 7;}
    size_t levelmax() const {return 7;}
    
    void Setup();

    size_t size(size_t i) const { return sizes_[i]; }

    const bounding_box<size_t>& get_global_range( void ) const
    {
        return global_range_;
    }

    void zero()
    {
        #pragma omp parallel for
        for (size_t i = 0; i < ntot_; ++i)
            data_[i] = 0.0;
    }

    data_t &relem(size_t i, size_t j, size_t k)
    {
        size_t idx = (i * sizes_[1] + j) * sizes_[3] + k;
        return data_[idx];
    }

    const data_t &relem(size_t i, size_t j, size_t k) const
    {
        size_t idx = (i * sizes_[1] + j) * sizes_[3] + k;
        return data_[idx];
    }

    ccomplex_t &kelem(size_t i, size_t j, size_t k)
    {
        size_t idx = (i * sizes_[1] + j) * sizes_[3] + k;
        return cdata_[idx];
    }

    const ccomplex_t &kelem(size_t i, size_t j, size_t k) const
    {
        size_t idx = (i * sizes_[1] + j) * sizes_[3] + k;
        return cdata_[idx];
    }

    ccomplex_t &kelem(size_t idx) { return cdata_[idx]; }
    const ccomplex_t &kelem(size_t idx) const { return cdata_[idx]; }
    data_t &relem(size_t idx) { return data_[idx]; }
    const data_t &relem(size_t idx) const { return data_[idx]; }

    size_t get_idx(size_t i, size_t j, size_t k) const
    {
        return (i * sizes_[1] + j) * sizes_[3] + k;
    }

    template <typename ft>
    vec3<ft> get_r(const size_t i, const size_t j, const size_t k) const
    {
        vec3<ft> rr;

        rr[0] = real_t(i + local_0_start_) * dx_[0];
        rr[1] = real_t(j) * dx_[1];
        rr[2] = real_t(k) * dx_[2];

        return rr;
    }

    void cell_pos( int ilevel, size_t i, size_t j, size_t k, double* x ) const {
        x[0] = double(i+local_0_start_)/size(0);
        x[1] = double(j)/size(1);
        x[2] = double(k)/size(2);
    }

    size_t count_leaf_cells( int, int ) const {
        return n_[0]*n_[1]*n_[2];
    }

    template <typename ft>
    vec3<ft> get_k(const size_t i, const size_t j, const size_t k) const
    {
        vec3<ft> kk;

#if defined(USE_MPI)
        auto ip = i + local_1_start_;
        kk[0] = (real_t(j) - real_t(j > nhalf_[0]) * n_[0]) * kfac_[0];
        kk[1] = (real_t(ip) - real_t(ip > nhalf_[1]) * n_[1]) * kfac_[1];
#else
        kk[0] = (real_t(i) - real_t(i > nhalf_[0]) * n_[0]) * kfac_[0];
        kk[1] = (real_t(j) - real_t(j > nhalf_[1]) * n_[1]) * kfac_[1];
#endif
        kk[2] = (real_t(k) - real_t(k > nhalf_[2]) * n_[2]) * kfac_[2];

        return kk;
    }

    Grid_FFT<data_t>& operator*=( data_t x ){
        if( space_ == kspace_id){
            this->apply_function_k( [&]( ccomplex_t& f ){ return f*x; } );
        }else{
            this->apply_function_r( [&]( data_t& f ){ return f*x; } );
        }
        return *this;
    }

    Grid_FFT<data_t>& operator/=( data_t x ){
        if( space_ == kspace_id){
            this->apply_function_k( [&]( ccomplex_t& f ){ return f/x; } );
        }else{
            this->apply_function_r( [&]( data_t& f ){ return f/x; } );
        }
        return *this;
    }

    Grid_FFT<data_t>& apply_Laplacian( void ){
        this->FourierTransformForward();
        this->apply_function_k_dep([&](auto x, auto k) {
            real_t kmod2 = k.norm_squared();
            return -x*kmod2;
        });
        this->zero_DC_mode();
        return *this;
    }

    Grid_FFT<data_t>& apply_InverseLaplacian( void ){
        this->FourierTransformForward();
        this->apply_function_k_dep([&](auto x, auto k) {
            real_t kmod2 = k.norm_squared();
            return -x/kmod2;
        });
        this->zero_DC_mode();
        return *this;
    }

    template <typename functional>
    void apply_function_k(const functional &f)
    {
#pragma omp parallel for
        for (size_t i = 0; i < sizes_[0]; ++i)
        {
            for (size_t j = 0; j < sizes_[1]; ++j)
            {
                for (size_t k = 0; k < sizes_[2]; ++k)
                {
                    auto &elem = this->kelem(i, j, k);
                    elem = f(elem);
                }
            }
        }
    }

    template <typename functional>
    void apply_function_r(const functional &f)
    {
#pragma omp parallel for
        for (size_t i = 0; i < sizes_[0]; ++i)
        {
            for (size_t j = 0; j < sizes_[1]; ++j)
            {
                for (size_t k = 0; k < sizes_[2]; ++k)
                {
                    auto &elem = this->relem(i, j, k);
                    elem = f(elem);
                }
            }
        }
    }

    double compute_2norm(void)
    {
        real_t sum1{0.0};
#pragma omp parallel for reduction(+ : sum1)
        for (size_t i = 0; i < sizes_[0]; ++i)
        {
            for (size_t j = 0; j < sizes_[1]; ++j)
            {
                for (size_t k = 0; k < sizes_[2]; ++k)
                {
                    const auto re = std::real(this->relem(i, j, k));
                    const auto im = std::imag(this->relem(i, j, k));
                    sum1 += re * re + im * im;
                }
            }
        }

        sum1 /= sizes_[0] * sizes_[1] * sizes_[2];

        return sum1;
    }

    double std(void)
    {
        real_t sum1{0.0}, sum2{0.0};
#pragma omp parallel for reduction(+ : sum1, sum2)
        for (size_t i = 0; i < sizes_[0]; ++i)
        {
            for (size_t j = 0; j < sizes_[1]; ++j)
            {
                for (size_t k = 0; k < sizes_[2]; ++k)
                {
                    const auto elem = std::real(this->relem(i, j, k));
                    sum1 += elem;
                    sum2 += elem * elem;
                }
            }
        }

        sum1 /= sizes_[0] * sizes_[1] * sizes_[2];
        sum2 /= sizes_[0] * sizes_[1] * sizes_[2];

        return std::sqrt(sum2 - sum1 * sum1);
    }
    double mean(void)
    {
        real_t sum1{0.0};
#pragma omp parallel for reduction(+ : sum1)
        for (size_t i = 0; i < sizes_[0]; ++i)
        {
            for (size_t j = 0; j < sizes_[1]; ++j)
            {
                for (size_t k = 0; k < sizes_[2]; ++k)
                {
                    const auto elem = std::real(this->relem(i, j, k));
                    sum1 += elem;
                }
            }
        }

        sum1 /= sizes_[0] * sizes_[1] * sizes_[2];

        return sum1;
    }

    template <typename functional, typename grid_t>
    void assign_function_of_grids_r(const functional &f, const grid_t &g)
    {
        assert(g.size(0) == size(0) && g.size(1) == size(1)); // && g.size(2) == size(2) );

#pragma omp parallel for
        for (size_t i = 0; i < sizes_[0]; ++i)
        {
            for (size_t j = 0; j < sizes_[1]; ++j)
            {
                for (size_t k = 0; k < sizes_[2]; ++k)
                {
                    auto &elem = this->relem(i, j, k);
                    const auto &elemg = g.relem(i, j, k);

                    elem = f(elemg);
                }
            }
        }
    }

    template <typename functional, typename grid1_t, typename grid2_t>
    void assign_function_of_grids_r(const functional &f, const grid1_t &g1, const grid2_t &g2)
    {
        assert(g1.size(0) == size(0) && g1.size(1) == size(1)); // && g1.size(2) == size(2));
        assert(g2.size(0) == size(0) && g2.size(1) == size(1)); // && g2.size(2) == size(2));

#pragma omp parallel for
        for (size_t i = 0; i < sizes_[0]; ++i)
        {
            for (size_t j = 0; j < sizes_[1]; ++j)
            {
                for (size_t k = 0; k < sizes_[2]; ++k)
                {
                    //auto idx = this->get_idx(i,j,k);
                    auto &elem = this->relem(i, j, k);

                    const auto &elemg1 = g1.relem(i, j, k);
                    const auto &elemg2 = g2.relem(i, j, k);

                    elem = f(elemg1, elemg2);
                }
            }
        }
    }

    template <typename functional, typename grid1_t, typename grid2_t, typename grid3_t>
    void assign_function_of_grids_r(const functional &f, const grid1_t &g1, const grid2_t &g2, const grid3_t &g3)
    {
        assert(g1.size(0) == size(0) && g1.size(1) == size(1)); // && g1.size(2) == size(2));
        assert(g2.size(0) == size(0) && g2.size(1) == size(1)); // && g2.size(2) == size(2));
        assert(g3.size(0) == size(0) && g3.size(1) == size(1)); // && g3.size(2) == size(2));

#pragma omp parallel for
        for (size_t i = 0; i < sizes_[0]; ++i)
        {
            for (size_t j = 0; j < sizes_[1]; ++j)
            {
                for (size_t k = 0; k < sizes_[2]; ++k)
                {
                    //auto idx = this->get_idx(i,j,k);
                    auto &elem = this->relem(i, j, k);

                    const auto &elemg1 = g1.relem(i, j, k);
                    const auto &elemg2 = g2.relem(i, j, k);
                    const auto &elemg3 = g3.relem(i, j, k);

                    elem = f(elemg1, elemg2, elemg3);
                }
            }
        }
    }

    template <typename functional>
    void apply_function_k_dep(const functional &f)
    {
#pragma omp parallel for
        for (size_t i = 0; i < sizes_[0]; ++i)
        {
            for (size_t j = 0; j < sizes_[1]; ++j)
            {
                for (size_t k = 0; k < sizes_[2]; ++k)
                {
                    auto &elem = this->kelem(i, j, k);
                    elem = f(elem, this->get_k<real_t>(i, j, k));
                }
            }
        }
    }

    template <typename functional>
    void apply_function_r_dep(const functional &f)
    {
#pragma omp parallel for
        for (size_t i = 0; i < sizes_[0]; ++i)
        {
            for (size_t j = 0; j < sizes_[1]; ++j)
            {
                for (size_t k = 0; k < sizes_[2]; ++k)
                {
                    auto &elem = this->relem(i, j, k);
                    elem = f(elem, this->get_r<real_t>(i, j, k));
                }
            }
        }
    }

    void FourierTransformBackward(bool do_transform = true);

    void FourierTransformForward(bool do_transform = true);

    void ApplyNorm(void);

    void FillRandomReal(unsigned long int seed = 123456ul);

    void Write_to_HDF5(std::string fname, std::string datasetname);

    void Write_PowerSpectrum( std::string ofname );

    void Compute_PowerSpectrum(std::vector<double> &bin_k, std::vector<double> &bin_P, std::vector<double> &bin_eP, std::vector<size_t> &bin_count, int nbins);

    void Write_PDF(std::string ofname, int nbins = 1000, double scale = 1.0, double rhomin = 1e-3, double rhomax = 1e3);

    void zero_DC_mode(void)
    {
        if( space_ == kspace_id ){
        #ifdef USE_MPI
        if (CONFIG::MPI_task_rank == 0)
        #endif
            cdata_[0] = (data_t)0.0;
        }else{
            data_t sum = 0.0;
// #pragma omp parallel for reduction(+:sum)
            for (size_t i = 0; i < sizes_[0]; ++i)
            {
                for (size_t j = 0; j < sizes_[1]; ++j)
                {
                    for (size_t k = 0; k < sizes_[2]; ++k)
                    {
                        sum += this->relem(i, j, k);
                    }
                }
            }
            #if defined(USE_MPI)
            data_t glob_sum = 0.0;
            MPI_Allreduce(reinterpret_cast<void *>(&sum), reinterpret_cast<void *>(&glob_sum),
                  1, GetMPIDatatype<data_t>(), MPI_SUM, MPI_COMM_WORLD);
            sum = glob_sum;
            #endif
            sum /= sizes_[0]*sizes_[1]*sizes_[2];

#pragma omp parallel for 
            for (size_t i = 0; i < sizes_[0]; ++i)
            {
                for (size_t j = 0; j < sizes_[1]; ++j)
                {
                    for (size_t k = 0; k < sizes_[2]; ++k)
                    {
                        this->relem(i, j, k) -= sum;
                    }
                }
            }
        }
    }
};
