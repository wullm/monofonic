/*******************************************************************\
 particle_generator.hh - This file is part of MUSIC2 -
 a code to generate initial conditions for cosmological simulations 
 
 CHANGELOG (only majors, for details see repo):
    10/2019 - Oliver Hahn - first implementation
\*******************************************************************/
#pragma once

#include <math/vec3.hh>
#include <grid_interpolate.hh>

#if defined(USE_HDF5)
#include "HDF_IO.hh"
#endif

namespace particle
{
    using vec3 = std::array<real_t,3>;

    enum lattice
    {
        lattice_glass = -1,
        lattice_sc = 0,  // SC : simple cubic
        lattice_bcc = 1, // BCC: body-centered cubic
        lattice_fcc = 2, // FCC: face-centered cubic
        lattice_rsc = 3, // RSC: refined simple cubic
    };

    const std::vector<std::vector<vec3_t<real_t>>> lattice_shifts =
        {
            // first shift must always be zero! (otherwise set_positions and set_velocities break)
            /* SC : */ {{0.0, 0.0, 0.0}},
            /* BCC: */ {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}},
            /* FCC: */ {{0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}},
            /* RSC: */ {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.5}, {0.0, 0.5, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}, {0.5, 0.5, 0.5}},
    };

    const std::vector<vec3_t<real_t>> second_lattice_shift =
        {
            /* SC : */ {0.5, 0.5, 0.5}, // this corresponds to CsCl lattice
            /* BCC: */ {0.5, 0.5, 0.0}, // is there a diatomic lattice with BCC base?!?
            /* FCC: */ {0.5, 0.5, 0.5}, // this corresponds to NaCl lattice
                                        // /* FCC: */ {0.25, 0.25, 0.25}, // this corresponds to Zincblende/GaAs lattice
            /* RSC: */ {0.25, 0.25, 0.25},
    };

    template <typename field_t>
    class lattice_generator
    {
        protected:

        struct glass
        {
            using data_t = typename field_t::data_t;
            size_t num_p, off_p;
            grid_interpolate<1, field_t> interp_;
            std::vector<vec3> glass_posr;

            glass( config_file& cf, const field_t &field )
            : num_p(0), off_p(0), interp_( field )
            {
                std::vector<real_t> glass_pos;
                real_t lglassbox = 1.0;

                std::string glass_fname = cf.get_value<std::string>("setup", "GlassFileName");
                size_t ntiles = cf.get_value<size_t>("setup", "GlassTiles");

#if defined(USE_HDF5)
                HDFReadGroupAttribute(glass_fname, "Header", "BoxSize", lglassbox);
                HDFReadDataset(glass_fname, "/PartType1/Coordinates", glass_pos);
#else
                throw std::runtime_error("Class lattice requires HDF5 support. Enable and recompile.");
#endif

                size_t np_in_file = glass_pos.size() / 3;
#if defined(USE_MPI)
                num_p = np_in_file * ntiles * ntiles * ntiles / MPI::get_size();
                off_p = MPI::get_rank() * num_p;
#else
                num_p = np_in_file * ntiles * ntiles * ntiles;
                off_p = 0;
#endif

                music::ilog << "Glass file contains " << np_in_file << " particles." << std::endl;

                glass_posr.assign(num_p, {0.0, 0.0, 0.0});

                std::array<real_t, 3> ng({real_t(field.n_[0]), real_t(field.n_[1]), real_t(field.n_[2])});

                #pragma omp parallel for
                for (size_t i = 0; i < num_p; ++i)
                {
                    size_t idxpart = off_p + i;
                    size_t idx_in_glass = idxpart % np_in_file;
                    size_t idxtile = idxpart / np_in_file;
                    size_t tile_z = idxtile % (ntiles * ntiles);
                    size_t tile_y = ((idxtile - tile_z) / ntiles) % ntiles;
                    size_t tile_x = (((idxtile - tile_z) / ntiles) - tile_y) / ntiles;
                    glass_posr[i][0] = std::fmod((glass_pos[3 * idx_in_glass + 0] / lglassbox + real_t(tile_x)) / ntiles * ng[0] + ng[0], ng[0]);
                    glass_posr[i][1] = std::fmod((glass_pos[3 * idx_in_glass + 1] / lglassbox + real_t(tile_y)) / ntiles * ng[1] + ng[1], ng[1]);
                    glass_posr[i][2] = std::fmod((glass_pos[3 * idx_in_glass + 2] / lglassbox + real_t(tile_z)) / ntiles * ng[2] + ng[2], ng[2]);
                }

#if defined(USE_MPI)
                interp_.domain_decompose_pos(glass_posr);

                num_p = glass_posr.size();
                std::vector<size_t> all_num_p( MPI::get_size(), 0 );
                MPI_Allgather( &num_p, 1, MPI_UNSIGNED_LONG_LONG, &all_num_p[0], 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD );
                off_p = 0;
                for( int itask=0; itask<=MPI::get_rank(); ++itask ){
                    off_p += all_num_p[itask];
                }
#endif
            }

            void update_ghosts( const field_t &field )
            {
                interp_.update_ghosts( field );
            }

            data_t get_at( const vec3& x ) const noexcept
            {
                return interp_.get_cic_at( x );
            }

            size_t size() const noexcept
            {
                return num_p;
            }

            size_t offset() const noexcept
            {
                return off_p;
            }
        };

        std::unique_ptr<glass> glass_ptr_;

        private:
        particle::container particles_;

        public:
        lattice_generator(lattice lattice_type, const bool b64reals, const bool b64ids, const size_t IDoffset, const field_t &field, config_file &cf)
        {
            if (lattice_type != lattice_glass)
            {
                // number of modes present in the field
                const size_t num_p_in_load = field.local_size();
                // unless SC lattice is used, particle number is a multiple of the number of modes (=num_p_in_load):
                const size_t overload = 1ull << std::max<int>(0, lattice_type); // 1 for sc, 2 for bcc, 4 for fcc, 8 for rsc
                // allocate memory for all local particles
                particles_.allocate(overload * num_p_in_load, b64reals, b64ids);
                // set particle IDs to the Lagrangian coordinate (1D encoded) with additionally the field shift encoded as well

                for (size_t i = 0, ipcount = 0; i < field.size(0); ++i)
                {
                    for (size_t j = 0; j < field.size(1); ++j)
                    {
                        for (size_t k = 0; k < field.size(2); ++k, ++ipcount)
                        {
                            for (size_t iload = 0; iload < overload; ++iload)
                            {
                                if (b64ids)
                                {
                                    particles_.set_id64(ipcount + iload * num_p_in_load, IDoffset + overload * field.get_cell_idx_1d(i, j, k) + iload);
                                }
                                else
                                {
                                    particles_.set_id32(ipcount + iload * num_p_in_load, IDoffset + overload * field.get_cell_idx_1d(i, j, k) + iload);
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                glass_ptr_ = std::make_unique<glass>( cf, field );
                particles_.allocate(glass_ptr_->size(), b64reals, b64ids);

                #pragma omp parallel for
                for (size_t i = 0; i < glass_ptr_->size(); ++i)
                {
                    if (b64ids)
                    {
                        particles_.set_id64(i, IDoffset + i + glass_ptr_->offset());
                    }
                    else
                    {
                        particles_.set_id32(i, IDoffset + i + glass_ptr_->offset());
                    }
                }
            }
        }

        // invalidates field, phase shifted to unspecified position after return
        void set_positions(const lattice lattice_type, bool is_second_lattice, int idim, real_t lunit, const bool b64reals, field_t &field, config_file &cf)
        {
            // works only for Bravais types
            if (lattice_type >= 0)
            {
                const size_t num_p_in_load = field.local_size();
                for (int ishift = 0; ishift < (1 << lattice_type); ++ishift)
                {
                    // if we are dealing with the secondary lattice, apply a global shift
                    if (ishift == 0 && is_second_lattice)
                    {
                        field.shift_field(second_lattice_shift[lattice_type]);
                    }

                    // can omit first shift since zero by convention, unless shifted already above, otherwise apply relative phase shift
                    if (ishift > 0)
                    {
                        field.shift_field(lattice_shifts[lattice_type][ishift] - lattice_shifts[lattice_type][ishift - 1]);
                    }
                    // read out values from phase shifted field and set assoc. particle's value
                    const auto ipcount0 = ishift * num_p_in_load;
                    for (size_t i = 0, ipcount = ipcount0; i < field.size(0); ++i)
                    {
                        for (size_t j = 0; j < field.size(1); ++j)
                        {
                            for (size_t k = 0; k < field.size(2); ++k)
                            {
                                auto pos = field.template get_unit_r_shifted<real_t>(i, j, k, lattice_shifts[lattice_type][ishift] + (is_second_lattice ? second_lattice_shift[lattice_type] : vec3_t<real_t>{0., 0., 0.}));
                                if (b64reals)
                                {
                                    particles_.set_pos64(ipcount++, idim, pos[idim] * lunit + field.relem(i, j, k));
                                }
                                else
                                {
                                    particles_.set_pos32(ipcount++, idim, pos[idim] * lunit + field.relem(i, j, k));
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                glass_ptr_->update_ghosts( field );
                #pragma omp parallel for
                for (size_t i = 0; i < glass_ptr_->size(); ++i)
                {
                    auto pos = glass_ptr_->glass_posr[i];
                    real_t disp = glass_ptr_->get_at(pos);
                    if (b64reals)
                    {
                        particles_.set_pos64(i, idim, pos[idim] / field.n_[idim] * lunit + disp);
                    }
                    else
                    {
                        particles_.set_pos32(i, idim, pos[idim] / field.n_[idim] * lunit + disp);
                    }
                }
            }
        }

        void set_velocities(lattice lattice_type, bool is_second_lattice, int idim, const bool b64reals, field_t &field, config_file &cf)
        {
            // works only for Bravais types
            if (lattice_type >= 0)
            {
                const size_t num_p_in_load = field.local_size();
                for (int ishift = 0; ishift < (1 << lattice_type); ++ishift)
                {
                    // if we are dealing with the secondary lattice, apply a global shift
                    if (ishift == 0 && is_second_lattice)
                    {
                        field.shift_field(second_lattice_shift[lattice_type]);
                    }
                    // can omit first shift since zero by convention, unless shifted already above, otherwise apply relative phase shift
                    if (ishift > 0)
                    {
                        field.shift_field(lattice_shifts[lattice_type][ishift] - lattice_shifts[lattice_type][ishift - 1]);
                    }
                    // read out values from phase shifted field and set assoc. particle's value
                    const auto ipcount0 = ishift * num_p_in_load;
                    for (size_t i = 0, ipcount = ipcount0; i < field.size(0); ++i)
                    {
                        for (size_t j = 0; j < field.size(1); ++j)
                        {
                            for (size_t k = 0; k < field.size(2); ++k)
                            {
                                if (b64reals)
                                {
                                    particles_.set_vel64(ipcount++, idim, field.relem(i, j, k));
                                }
                                else
                                {
                                    particles_.set_vel32(ipcount++, idim, field.relem(i, j, k));
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                glass_ptr_->update_ghosts( field );
                #pragma omp parallel for
                for (size_t i = 0; i < glass_ptr_->size(); ++i)
                {
                    auto pos = glass_ptr_->glass_posr[i];
                    real_t vel = glass_ptr_->get_at(pos);
                    if (b64reals)
                    {
                        particles_.set_vel64(i, idim, vel);
                    }
                    else
                    {
                        particles_.set_vel32(i, idim, vel);
                    }
                }
            }
        }

        const particle::container& get_particles() const noexcept{
            return particles_;
        }

    }; // struct lattice

} // namespace particle
