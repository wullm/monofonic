#include <unistd.h> // for unlink
#include <sys/types.h>
#include <sys/stat.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>

#include <general.hh>
#include <output_plugin.hh>

//! Implementation of class grafic2_output_plugin
/*!
 This class implements a grafic-2 (cf. Bertschinger 2001) compatible
 output format. With some RAMSES extras.
*/
class grafic2_output_plugin : public output_plugin
{
private:
    struct header
    {
        int n1, n2, n3;
        float dxini0;
        float xoff10, xoff20, xoff30;
        float astart0, omega_m0, omega_l0, h00;
    };

    std::string get_file_name(const cosmo_species &s, const fluid_component &c) const;
    void write_ramses_namelist(void) const;

protected:
    header header_;
    real_t lunit_, vunit_;
    uint32_t levelmin_;
    bool bhavebaryons_;
    std::vector<float> data_buf_;
    std::string dirname_;

public:
    //! constructor
    explicit grafic2_output_plugin(ConfigFile &cf)
        : output_plugin(cf, "GRAFIC2/RAMSES")
    {
        lunit_ = 1.0;
        vunit_ = 1.0;

        double
            boxlength = cf_.GetValue<double>("setup", "BoxLength"),
            H0 = cf_.GetValue<double>("cosmology", "H0"),
            zstart = cf_.GetValue<double>("setup", "zstart"),
            astart = 1.0 / (1.0 + zstart),
            omegam = cf_.GetValue<double>("cosmology", "Omega_m"),
            omegaL = cf_.GetValue<double>("cosmology", "Omega_L");
        uint32_t ngrid = cf_.GetValue<int>("setup", "GridRes");


        levelmin_ = uint32_t( std::log2( double(ngrid) ) + 1e-6 );

        if( std::abs( std::pow( 2.0, levelmin_ )-double(ngrid) ) > 1e-4 ){
            csoca::elog << interface_name_ << " plugin requires setup/GridRes to be power of 2!" << std::endl;
            abort();
        }

        bhavebaryons_ = cf_.GetValueSafe<bool>("setup", "baryons", false);

        header_.n1 = ngrid;
        header_.n2 = ngrid;
        header_.n3 = ngrid;
        header_.dxini0 = boxlength / (H0 * 0.01) / ngrid;
        header_.xoff10 = 0;
        header_.xoff20 = 0;
        header_.xoff30 = 0;
        header_.astart0 = astart;
        header_.omega_m0 = omegam;
        header_.omega_l0 = omegaL;
        header_.h00 = H0;

        data_buf_.assign(ngrid * ngrid, 0.0f);

        lunit_ = boxlength;
        vunit_ = boxlength;

        // create directory structure
        dirname_ = this->fname_;
        remove(dirname_.c_str());
        mkdir(dirname_.c_str(), 0777);

        // write RAMSES namelist file?
        if (cf_.GetValueSafe<bool>("output", "ramses_nml", true))
        {
            write_ramses_namelist();
        }
    }

    bool write_species_as_grid(const cosmo_species &) { return true; }

    bool write_species_as_particles(const cosmo_species &) { return false; }

    real_t position_unit() const { return lunit_; }

    real_t velocity_unit() const { return vunit_; }

    void write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c);
};

std::string grafic2_output_plugin::get_file_name(const cosmo_species &s, const fluid_component &c) const
{
    std::string file_name, species_str;

    file_name = dirname_ + "/ic_";

    switch (s)
    {
    case cosmo_species::dm:
        species_str = "c";
        break;
    case cosmo_species::baryon:
        species_str = "b";
        break;
    case cosmo_species::neutrino:
        species_str = "n";
        break;
    default:
        break;
    }

    switch (c)
    {
    case fluid_component::density:
        file_name += "delta" + species_str;
        break;
    case fluid_component::vx:
        file_name += "vel" + species_str + "x";
        break;
    case fluid_component::vy:
        file_name += "vel" + species_str + "y";
        break;
    case fluid_component::vz:
        file_name += "vel" + species_str + "z";
        break;
    case fluid_component::dx:
        file_name += "pos" + species_str + "x";
        break;
    case fluid_component::dy:
        file_name += "pos" + species_str + "y";
        break;
    case fluid_component::dz:
        file_name += "pos" + species_str + "z";
        break;
    default:
        break;
    }

    return file_name;
}

void grafic2_output_plugin::write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c)
{
    std::string file_name = this->get_file_name(s, c);

    for (int write_rank = 0; write_rank < CONFIG::MPI_task_size; ++write_rank)
    {
        if (write_rank == CONFIG::MPI_task_rank)
        {
            // std::cerr << "WRITING..." << std::endl;
            if (write_rank == 0)
            {
                unlink(file_name.c_str());
            }
            std::ofstream ofs(file_name.c_str(), std::ios::binary);

            // write header or seek to end of file
            if (write_rank == 0)
            {
                uint32_t blocksz = sizeof(header);
                ofs.write(reinterpret_cast<const char *>(&blocksz), sizeof(int));
                ofs.write(reinterpret_cast<const char *>(&header_), blocksz);
                ofs.write(reinterpret_cast<const char *>(&blocksz), sizeof(int));
            }
            else
            {
                // seek to end of file
                ofs.seekp(std::ios::end);
            }

            // check field size against buffer size...
            uint32_t ngrid = cf_.GetValue<int>("setup", "GridRes");
            assert(g.size(0) == ngrid && g.size(1) == ngrid && g.size(2) == ngrid);

            // write actual field slice by slice
            for (size_t i = 0; i < g.size(0); ++i)
            {
                for (unsigned j = 0; j < g.size(1); ++j)
                {
                    for (unsigned k = 0; k < g.size(2); ++k)
                    {
                        data_buf_[j * ngrid + k] = g.relem(i, j, k);
                    }
                }

                uint32_t blocksz = ngrid * ngrid * sizeof(float);
                ofs.write(reinterpret_cast<const char *>(&blocksz), sizeof(uint32_t));
                ofs.write(reinterpret_cast<const char *>(&data_buf_[0]), blocksz);
                ofs.write(reinterpret_cast<const char *>(&blocksz), sizeof(uint32_t));
            }

            ofs.close();
        }

        multitask_sync_barrier();

    } // end loop over write_rank

    csoca::ilog << interface_name_ << " : Wrote field to file \'" << file_name << "\'" << std::endl;
}

void grafic2_output_plugin::write_ramses_namelist(void) const
{
    //... also write the refinement options to a dummy namelist file
    char ff[256];
    sprintf(ff, "%s/ramses.nml", dirname_.c_str());

    std::ofstream ofst(ff, std::ios::trunc);

    // -- RUN_PARAMS -- //
    ofst
        << "&RUN_PARAMS\n"
        << "cosmo=.true.\n"
        << "pic=.true.\n"
        << "poisson=.true.\n";

    if (bhavebaryons_)
        ofst << "hydro=.true.\n";
    else
        ofst << "hydro=.false.\n";

    ofst
        << "nrestart=0\n"
        << "nremap=1\n"
        << "nsubcycle=1,2\n"
        << "ncontrol=1\n"
        << "verbose=.false.\n"
        << "/\n\n";

    // -- INIT_PARAMS -- //
    ofst
        << "&INIT_PARAMS\n"
        << "filetype=\'grafic\'\n"
        << "initfile(1)=\'" << dirname_ << "\'\n"
        << "/\n\n";

    // initialize with settings for naddref additional levels of refinement
    unsigned naddref = 5; 

    // -- AMR_PARAMS -- //
    ofst << "&AMR_PARAMS\n"
         << "levelmin=" << levelmin_ << "\n"
         << "levelmax=" << levelmin_ + naddref << "\n"
         << "nexpand=1";

    ofst << "\n"
         << "ngridmax=2000000\n"
         << "npartmax=3000000\n"
         << "/\n\n";

    ofst << "&REFINE_PARAMS\n"
         << "m_refine=" << 1 + naddref << "*8.,\n"
         << "/\n";

    csoca::ilog << interface_name_ << " wrote partial RAMSES namelist file \'" << fname_ << "\'" << std::endl;
}

namespace
{
output_plugin_creator_concrete<grafic2_output_plugin> creator1("grafic2");
} // namespace
