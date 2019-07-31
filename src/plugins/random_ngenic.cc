
#include <general.hh>
#include <random_plugin.hh>
#include <config_file.hh>

#include <vector>
#include <cmath>

#include <grid_fft.hh>
#include <gsl/gsl_rng.h>

class RNG_ngenic : public RNG_plugin
{
private:
    gsl_rng *pRandomGenerator_;
    long RandomSeed_;
    size_t nres_;
    std::vector<unsigned int> SeedTable_;

public:
    explicit RNG_ngenic(ConfigFile &cf) : RNG_plugin(cf)
    {

        RandomSeed_ = cf.GetValue<long>("random", "seed");
        nres_ = cf.GetValue<size_t>("setup", "GridRes");
        pRandomGenerator_ = gsl_rng_alloc(gsl_rng_ranlxd1);

        SeedTable_.assign(nres_ * nres_, 0u);

        for (size_t i = 0; i < nres_ / 2; ++i)
        {
            for (size_t j = 0; j < i; j++)
                SeedTable_[i * nres_ + j] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i + 1; j++)
                SeedTable_[j * nres_ + i] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i; j++)
                SeedTable_[(nres_ - 1 - i) * nres_ + j] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i + 1; j++)
                SeedTable_[(nres_ - 1 - j) * nres_ + i] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i; j++)
                SeedTable_[i * nres_ + (nres_ - 1 - j)] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i + 1; j++)
                SeedTable_[j * nres_ + (nres_ - 1 - i)] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i; j++)
                SeedTable_[(nres_ - 1 - i) * nres_ + (nres_ - 1 - j)] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);

            for (size_t j = 0; j < i + 1; j++)
                SeedTable_[(nres_ - 1 - j) * nres_ + (nres_ - 1 - i)] = 0x7fffffff * gsl_rng_uniform(pRandomGenerator_);
        }
    }

    virtual ~RNG_ngenic()
    {
        gsl_rng_free(pRandomGenerator_);
    }

    bool isMultiscale() const { return false; }

    void Fill_Grid(Grid_FFT<real_t> &g) const
    {
        g.FourierTransformForward(false);

        // transform is transposed!
        for (size_t i = 0; i < nres_; ++i) 
        {
            size_t ii  = (i>0)? nres_ - i : 0;
            size_t ip  = i - g.local_1_start_;
            size_t iip = ii- g.local_1_start_;
            bool i_in_range  = (i >= size_t(g.local_1_start_) && i < size_t(g.local_1_start_+g.local_1_size_));
            bool ii_in_range = (ii >= size_t(g.local_1_start_) && ii < size_t(g.local_1_start_ + g.local_1_size_));

            if( i_in_range || ii_in_range )
            {
                for (size_t j = 0; j < nres_; ++j) 
                {                   
                    ptrdiff_t jj = (j>0)? nres_ - j : 0;
                    gsl_rng_set( pRandomGenerator_, SeedTable_[i * nres_ + j]);
                    for (size_t k = 0; k < g.size(2); ++k) 
                    {
                        double phase = gsl_rng_uniform(pRandomGenerator_) * 2 * M_PI;
                        double ampl = 0;

                        do {
                            ampl = gsl_rng_uniform(pRandomGenerator_);
                        } while (ampl == 0||ampl == 1);

                        if (i == nres_ / 2 || j == nres_ / 2 || k == nres_ / 2) continue;
                        if (i == 0 && j == 0 && k == 0) continue;

                        ampl = -std::sqrt(-std::log(ampl));
                        ccomplex_t zrand(ampl*std::cos(phase),ampl*std::sin(phase));

                        if (k > 0) {
                            if (i_in_range) g.kelem(ip,j,k) = zrand;
                        } else{ /* k=0 plane needs special treatment */
                            if (i == 0) {
                                if (j < nres_ / 2 && i_in_range)
                                {
                                    g.kelem(ip,j,k) = zrand;
                                    g.kelem(ip,jj,k) = std::conj(zrand);
                                }
                            } else if (i < nres_ / 2) {
                                if(i_in_range) g.kelem(ip,j,k) = zrand;
                                if (ii_in_range) g.kelem(iip,jj,k) = std::conj(zrand);
                            }
                        }
                    }
                }
            }
        }
    }
};

namespace
{
RNG_plugin_creator_concrete<RNG_ngenic> creator("NGENIC");
}
