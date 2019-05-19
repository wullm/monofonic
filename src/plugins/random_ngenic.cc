
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
        double fnorm = std::pow((double)nres_, -1.5);
        g.FourierTransformForward(false);

#ifdef USE_MPI
        // transform is transposed!
        for (size_t j = 0; j < g.size(0); ++j)
        {
            for (size_t i = 0; i < g.size(1); ++i)
            {
#else
        for (size_t i = 0; i < g.size(0); ++i)
        {
            for (size_t j = 0; j < g.size(1); ++j)
            {
#endif
                ptrdiff_t ii = (i>0)? g.size(1) - i : 0;
                gsl_rng_set( pRandomGenerator_, SeedTable_[i * nres_ + j]);
                for (size_t k = 0; k < g.size(2); ++k)
                {
                    double phase = gsl_rng_uniform(pRandomGenerator_) * 2 * M_PI;
                    double ampl;
                    do
                    {
                        ampl = gsl_rng_uniform(pRandomGenerator_);
                    } while (ampl == 0);
                    if (i == nres_ / 2 || j == nres_ / 2 || k == nres_ / 2)
                        continue;
                    if (i == 0 && j == 0 && k == 0)
                        continue;

                    real_t rp = -std::sqrt(-std::log(ampl)) * std::cos(phase);// * fnorm;
                    real_t ip = -std::sqrt(-std::log(ampl)) * std::sin(phase);// * fnorm;
                    ccomplex_t zrand(rp,ip);

                    if (k > 0)
                    {
                        g.kelem(i,j,k) = zrand;
                        // RE(knoise[(i * res + j) * (res / 2 + 1) + k]) = rp;
                        // IM(knoise[(i * res + j) * (res / 2 + 1) + k]) = ip;
                    }
                    else /* k=0 plane needs special treatment */
                    {
                        if (i == 0)
                        {
                            if (j >= nres_ / 2)
                            {
                                continue;
                            }
                            else
                            {
                                int jj = (int)nres_ - (int)j; /* note: j!=0 surely holds at this point */
                                g.kelem(i,j,k) = zrand;
                                g.kelem(i,jj,k) = std::conj(zrand);

                                // RE(knoise[(i * res + j) * (res / 2 + 1) + k]) = rp;
                                // IM(knoise[(i * res + j) * (res / 2 + 1) + k]) = ip;

                                // RE(knoise[(i * res + jj) * (res / 2 + 1) + k]) = rp;
                                // IM(knoise[(i * res + jj) * (res / 2 + 1) + k]) = -ip;
                            }
                        }
                        else
                        {
                            if (i >= nres_ / 2)
                            {
                                continue;
                            }
                            else
                            {
                                ptrdiff_t ii = (i>0)? nres_ - i : 0;
                                ptrdiff_t jj = (j>0)? nres_ - j : 0;
                                
                                g.kelem(i,j,k) = zrand;

                                // RE(knoise[(i * res + j) * (res / 2 + 1) + k]) = rp;
                                // IM(knoise[(i * res + j) * (res / 2 + 1) + k]) = ip;

                                if (ii >= 0 && ii < (int)nres_)
                                {
                                    // RE(knoise[(ii * res + jj) * (res / 2 + 1) + k]) = rp;
                                    // IM(knoise[(ii * res + jj) * (res / 2 + 1) + k]) = -ip;
                                    g.kelem(ii,jj,k) = std::conj(zrand);
                                }
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
