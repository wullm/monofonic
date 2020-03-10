/*******************************************************************\
 particle_generator.hh - This file is part of MUSIC2 -
 a code to generate initial conditions for cosmological simulations 
 
 CHANGELOG (only majors, for details see repo):
    10/2019 - Oliver Hahn - first implementation
\*******************************************************************/
#pragma once

#include <vec3.hh>

namespace particle {

enum lattice{
    lattice_sc  = 0, // SC : simple cubic
    lattice_bcc = 1, // BCC: body-centered cubic
    lattice_fcc = 2, // FCC: face-centered cubic
    lattice_rsc = 3, // RSC: refined simple cubic
};

const std::vector< std::vector<vec3<real_t>> > lattice_shifts = 
{   
    // first shift must always be zero! (otherwise set_positions and set_velocities break)
    /* SC : */ {{0.0,0.0,0.0}},
    /* BCC: */ {{0.0,0.0,0.0},{0.5,0.5,0.5}},
    /* FCC: */ {{0.0,0.0,0.0},{0.0,0.5,0.5},{0.5,0.0,0.5},{0.5,0.5,0.0}},
    /* RSC: */ {{0.0,0.0,0.0},{0.0,0.0,0.5},{0.0,0.5,0.0},{0.0,0.5,0.5},{0.5,0.0,0.0},{0.5,0.0,0.5},{0.5,0.5,0.0},{0.5,0.5,0.5}},
};

const std::vector<vec3<real_t>> second_lattice_shift =
{
        /* SC : */ {0.5, 0.5, 0.5}, // this corresponds to CsCl lattice
        /* BCC: */ {0.5, 0.5, 0.0}, // is there a diatomic lattice with BCC base?!?
        /* FCC: */ {0.5, 0.5, 0.5}, // this corresponds to NaCl lattice
        // /* FCC: */ {0.25, 0.25, 0.25}, // this corresponds to Zincblende/GaAs lattice
        /* RSC: */ {0.25, 0.25, 0.25},
};

template<typename field_t>
void initialize_lattice( container& particles, lattice lattice_type, const bool b64reals, const bool b64ids, const size_t IDoffset, const field_t& field ){
    // number of modes present in the field
    const size_t num_p_in_load = field.local_size();
    // unless SC lattice is used, particle number is a multiple of the number of modes (=num_p_in_load):
    const size_t overload = 1ull<<lattice_type; // 1 for sc, 2 for bcc, 4 for fcc, 8 for rsc
    // allocate memory for all local particles
    particles.allocate( overload * num_p_in_load, b64reals, b64ids );
    // set particle IDs to the Lagrangian coordinate (1D encoded) with additionally the field shift encoded as well
    for( size_t i=0,ipcount=0; i<field.size(0); ++i ){
        for( size_t j=0; j<field.size(1); ++j){
            for( size_t k=0; k<field.size(2); ++k,++ipcount){
                for( size_t iload=0; iload<overload; ++iload ){
                    if( b64ids ){
                        particles.set_id64( ipcount+iload*num_p_in_load, IDoffset + overload*field.get_cell_idx_1d(i,j,k)+iload );
                    }else{
                        particles.set_id32( ipcount+iload*num_p_in_load, IDoffset + overload*field.get_cell_idx_1d(i,j,k)+iload );
                    }
                }
            }
        }
    }
}

// invalidates field, phase shifted to unspecified position after return
template<typename field_t>
void set_positions( container& particles, const lattice lattice_type, bool is_second_lattice, int idim, real_t lunit, const bool b64reals, field_t& field )
{
    const size_t num_p_in_load = field.local_size();
    for( int ishift=0; ishift<(1<<lattice_type); ++ishift ){
        // if we are dealing with the secondary lattice, apply a global shift
        if( ishift==0 && is_second_lattice ){
            field.shift_field( second_lattice_shift[lattice_type] );
        }

        // can omit first shift since zero by convention, unless shifted already above, otherwise apply relative phase shift
        if( ishift>0 ){
            field.shift_field( lattice_shifts[lattice_type][ishift] - lattice_shifts[lattice_type][ishift-1] );
        }
        // read out values from phase shifted field and set assoc. particle's value
        const auto ipcount0 = ishift * num_p_in_load;
        for( size_t i=0,ipcount=ipcount0; i<field.size(0); ++i ){
            for( size_t j=0; j<field.size(1); ++j){
                for( size_t k=0; k<field.size(2); ++k){
                    auto pos = field.template get_unit_r_shifted<real_t>(i,j,k,lattice_shifts[lattice_type][ishift] 
                        + (is_second_lattice? second_lattice_shift[lattice_type] : vec3<real_t>{0.,0.,0.}) );
                    if( b64reals ){
                        particles.set_pos64( ipcount++, idim, pos[idim]*lunit + field.relem(i,j,k) );
                    }else{
                        particles.set_pos32( ipcount++, idim, pos[idim]*lunit + field.relem(i,j,k) );
                    }
                }
            }
        }
    }
}

template <typename field_t>
void set_velocities(container &particles, lattice lattice_type, bool is_second_lattice, int idim, const bool b64reals, field_t &field)
{
    const size_t num_p_in_load = field.local_size();
    for( int ishift=0; ishift<(1<<lattice_type); ++ishift ){
        // if we are dealing with the secondary lattice, apply a global shift
        if (ishift == 0 && is_second_lattice){
            field.shift_field(second_lattice_shift[lattice_type]);
        }
        // can omit first shift since zero by convention, unless shifted already above, otherwise apply relative phase shift
        if (ishift > 0){
            field.shift_field( lattice_shifts[lattice_type][ishift]-lattice_shifts[lattice_type][ishift-1] );
        }
        // read out values from phase shifted field and set assoc. particle's value
        const auto ipcount0 = ishift * num_p_in_load;
        for( size_t i=0,ipcount=ipcount0; i<field.size(0); ++i ){
            for( size_t j=0; j<field.size(1); ++j){
                for( size_t k=0; k<field.size(2); ++k){
                    if( b64reals ){
                        particles.set_vel64( ipcount++, idim, field.relem(i,j,k) );
                    }else{
                        particles.set_vel32( ipcount++, idim, field.relem(i,j,k) );
                    }
                }
            }
        }
    }
}


} // end namespace particles
