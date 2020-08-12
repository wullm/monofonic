#ifdef ENABLE_GENERICIO

#include <output_plugin.hh>
#include <GenericIO.h>
#include <mpi.h>

class genericio_output_plugin : public output_plugin
{
protected:
    real_t lunit_, vunit_;
    bool hacc_hydro_;
    float hacc_etamax_;
    float hh_value_, rho_value_, mu_value_;
    std::vector<int64_t> ids;
    std::vector<float> xx, yy, zz;
    std::vector<float> vx, vy, vz;
    std::vector<float> mass, hh, uu;
    std::vector<float> mu, phi, rho;
    std::vector<int16_t> mask;

public:
    //! constructor
    explicit genericio_output_plugin(config_file &cf)
            : output_plugin(cf, "GenericIO")
    {
        real_t astart = 1.0 / (1.0 + cf_.get_value<double>("setup", "zstart"));
        lunit_ = cf_.get_value<double>("setup", "BoxLength");
        vunit_ = lunit_;
        hacc_hydro_ = cf_.get_value_safe<bool>("output", "GenericIO_HACCHydro", false);
        hacc_etamax_ = cf_.get_value_safe<float>("output", "GenericIO_ETAMAX", 1.0f);
        hh_value_ = 4.0f * hacc_etamax_ * lunit_ / cf_.get_value<float>("setup", "GridRes");
        mu_value_ = 4.0 / (8.0 - 5.0 * (1.0 - 0.75)); // neutral value. FIXME: account for ionization?
        
        double rhoc = 27.7519737; // in h^2 1e10 M_sol / Mpc^3
        rho_value_ = cf_.get_value<double>("cosmology", "Omega_b") * rhoc;
    }

    output_type write_species_as(const cosmo_species &) const { return output_type::particles; }

    real_t position_unit() const { return lunit_; }

    real_t velocity_unit() const { return vunit_; }

    bool has_64bit_reals() const { return false; }

    bool has_64bit_ids() const { return true; }

    void write_particle_data(const particle::container &pc, const cosmo_species &s, double Omega_species)
    {
        double rhoc = 27.7519737; // in h^2 1e10 M_sol / Mpc^3
        double boxmass = Omega_species * rhoc * std::pow(cf_.get_value<double>("setup", "BoxLength"), 3.);
        double particle_mass = boxmass / pc.get_global_num_particles();

        size_t npart = pc.get_local_num_particles();
        xx.reserve(xx.size() + npart);
        yy.reserve(yy.size() + npart);
        zz.reserve(zz.size() + npart);
        vx.reserve(vx.size() + npart);
        vy.reserve(vy.size() + npart);
        vz.reserve(vz.size() + npart);
        ids.reserve(ids.size() + npart);
        
        

        auto _pos = reinterpret_cast<const float*>(pc.get_pos32_ptr());
        auto _vel = reinterpret_cast<const float*>(pc.get_vel32_ptr());
        auto _ids = reinterpret_cast<const uint64_t*>(pc.get_ids64_ptr());
        
        for(size_t i=0; i<npart; ++i) {
            xx.push_back(_pos[3*i + 0]);
            yy.push_back(_pos[3*i + 1]);
            zz.push_back(_pos[3*i + 2]);
            vx.push_back(_vel[3*i + 0]);
            vy.push_back(_vel[3*i + 1]);
            vz.push_back(_vel[3*i + 2]);
        }

        // phi doesn't need to be initialized, just needs to be present in data
        phi.resize(phi.size() + npart);
        std::copy(_ids, _ids+npart, std::back_inserter(ids));

        if(hacc_hydro_) {
            size_t prev_size = mass.size();
            size_t new_size = prev_size + npart;
            
            mass.resize(new_size);
            std::fill(mass.begin() + prev_size, mass.end(), particle_mass);

            mask.resize(new_size);
            std::fill(mask.begin() + prev_size, mask.end(), s == cosmo_species::baryon ? 1<<2 : 0);

            hh.resize(new_size);
            std::fill(hh.begin() + prev_size, hh.end(), s == cosmo_species::baryon ? hh_value_ : 0.0f);

            uu.resize(new_size);
            std::fill(uu.begin() + prev_size, uu.end(), s == cosmo_species::baryon ? 0.0f : 0.0f);

            rho.resize(new_size);
            std::fill(rho.begin() + prev_size, rho.end(), s == cosmo_species::baryon ? rho_value_ : 0.0f);

            mu.resize(new_size);
            std::fill(mu.begin() + prev_size, mu.end(), s == cosmo_species::baryon ? mu_value_ : 0.0f);
        }
    }

    ~genericio_output_plugin() override {
        gio::GenericIO writer(MPI_COMM_WORLD, fname_, gio::GenericIO::FileIO::FileIOMPI);
        writer.setPhysOrigin(0., -1);
        writer.setPhysScale(lunit_, -1);
        writer.setNumElems(xx.size());
        writer.addVariable("x", xx);
        writer.addVariable("y", yy);
        writer.addVariable("z", zz);
        writer.addVariable("vx", vx);
        writer.addVariable("vy", vy);
        writer.addVariable("vz", vz);
        writer.addVariable("id", ids);
        writer.addVariable("phi", phi);
        if(hacc_hydro_) {
            writer.addVariable("mass", mass);
            writer.addVariable("mask", mask);
            writer.addVariable("hh", hh);
            writer.addVariable("uu", uu);
            writer.addVariable("rho", rho);
            writer.addVariable("mu", mu);
        }
        writer.write();
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

namespace
{
output_plugin_creator_concrete<genericio_output_plugin> creator("genericio");
} // namespace

#endif // ENABLE_GENERICIO