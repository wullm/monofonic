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
    /* FCC: */ {{0.0,0.0,0.0},{0.5,0.5,0.0},{0.5,0.0,0.5},{0.0,0.5,0.5}},
    /* RSC: */ {{0.0,0.0,0.0},{0.0,0.0,0.5},{0.0,0.5,0.0},{0.0,0.5,0.5},{0.5,0.0,0.0},{0.5,0.0,0.5},{0.5,0.5,0.0},{0.5,0.5,0.5}},
};

template<typename field_t>
void initialize_lattice( container& particles, lattice lattice_type, const field_t& field ){
    const size_t num_p_in_load = field.local_size();
    const size_t overload = 1ull<<lattice_type; // 1 for sc, 2 for bcc, 4 for fcc, 8 for rsc

    particles.allocate( overload * num_p_in_load );

    for( size_t i=0,ipcount=0; i<field.size(0); ++i ){
        for( size_t j=0; j<field.size(1); ++j){
            for( size_t k=0; k<field.size(2); ++k,++ipcount){
                for( size_t iload=0; iload<overload; ++iload ){
                    particles.set_id( ipcount+iload*num_p_in_load, overload*field.get_cell_idx_1d(i,j,k)+iload );
                }
            }
        }
    }
}

// invalidates field, phase shifted to unspecified position after return
template<typename field_t>
void set_positions( container& particles, const lattice lattice_type, int idim, real_t lunit, field_t& field )
{
    const size_t num_p_in_load = field.local_size();
    for( int ishift=0; ishift<(1<<lattice_type); ++ishift ){
        // can omit first shift since zero by convention, otherwise apply phase shift
        if( ishift>0 ){
            vec3<real_t> shift = lattice_shifts[lattice_type][ishift]-lattice_shifts[lattice_type][ishift-1];
            field.shift_field( shift.x, shift.y, shift.z );
        }
        auto ipcount0 = ishift * num_p_in_load;
        for( size_t i=0,ipcount=ipcount0; i<field.size(0); ++i ){
            for( size_t j=0; j<field.size(1); ++j){
                for( size_t k=0; k<field.size(2); ++k){
                    auto pos = field.template get_unit_r_shifted<real_t>(i,j,k,lattice_shifts[lattice_type][ishift]);
                    particles.set_pos( ipcount++, idim, pos[idim]*lunit + field.relem(i,j,k) );
                }
            }
        }
    }
}

template<typename field_t>
void set_velocities( container& particles, lattice lattice_type, int idim, field_t& field )
{
    const size_t num_p_in_load = field.local_size();
    for( int ishift=0; ishift<(1<<lattice_type); ++ishift ){
        // can omit first shift since zero by convention, otherwise apply phase shift
        if( ishift>0 ){
            vec3<real_t> shift = lattice_shifts[lattice_type][ishift]-lattice_shifts[lattice_type][ishift-1];
            field.shift_field( shift.x, shift.y, shift.z );
        }
        auto ipcount0 = ishift * num_p_in_load;
        for( size_t i=0,ipcount=ipcount0; i<field.size(0); ++i ){
            for( size_t j=0; j<field.size(1); ++j){
                for( size_t k=0; k<field.size(2); ++k){
                    particles.set_vel( ipcount++, idim, field.relem(i,j,k) );
                }
            }
        }
    }
}


///// deprecated code below ////////////////////////////////////////////////////

// invalidates field, phase shifted to unspecified position after return
template<typename field_t>
void set_positions_old( container& particles, lattice lattice_type, int idim, real_t lunit, field_t& field )
{
    const size_t num_p_in_load = field.local_size();

    for( size_t i=0,ipcount=0; i<field.size(0); ++i ){
        for( size_t j=0; j<field.size(1); ++j){
            for( size_t k=0; k<field.size(2); ++k){
                auto pos = field.template get_unit_r<real_t>(i,j,k);
                particles.set_pos( ipcount++, idim, pos[idim]*lunit + field.relem(i,j,k) );
            }
        }
    }

    if( lattice_type == particle::lattice_bcc ){
        field.shift_field( 0.5, 0.5, 0.5 );
        auto ipcount0 = num_p_in_load;
        for( size_t i=0,ipcount=ipcount0; i<field.size(0); ++i ){
            for( size_t j=0; j<field.size(1); ++j){
                for( size_t k=0; k<field.size(2); ++k){
                    auto pos = field.template get_unit_r_shifted<real_t>(i,j,k,0.5,0.5,0.5);
                    particles.set_pos( ipcount++, idim, pos[idim]*lunit + field.relem(i,j,k) );
                }
            }
        }
    }
    else if( lattice_type == particle::lattice_fcc ){ 
        // 0.5 0.5 0.0
        field.shift_field( 0.5, 0.5, 0.0 );
        auto ipcount0 = num_p_in_load;
        for( size_t i=0,ipcount=ipcount0; i<field.size(0); ++i ){
            for( size_t j=0; j<field.size(1); ++j){
                for( size_t k=0; k<field.size(2); ++k){
                    auto pos = field.template get_unit_r_shifted<real_t>(i,j,k,0.5,0.5,0.0);
                    particles.set_pos( ipcount++, idim, pos[idim]*lunit + field.relem(i,j,k) );
                }
            }
        }
        // 0.0 0.5 0.5
        field.shift_field( -0.5, 0.0, 0.5 );
        ipcount0 = 2*num_p_in_load;
        for( size_t i=0,ipcount=ipcount0; i<field.size(0); ++i ){
            for( size_t j=0; j<field.size(1); ++j){
                for( size_t k=0; k<field.size(2); ++k){
                    auto pos = field.template get_unit_r_shifted<real_t>(i,j,k,0.0,0.5,0.5);
                    particles.set_pos( ipcount++, idim, pos[idim]*lunit + field.relem(i,j,k) );
                }
            }
        }
        // 0.5 0.0 0.5
        field.shift_field( 0.5, -0.5, 0.0 );
        ipcount0 = 3*num_p_in_load;
        for( size_t i=0,ipcount=ipcount0; i<field.size(0); ++i ){
            for( size_t j=0; j<field.size(1); ++j){
                for( size_t k=0; k<field.size(2); ++k){
                    auto pos = field.template get_unit_r_shifted<real_t>(i,j,k,0.5,0.0,0.5);
                    particles.set_pos( ipcount++, idim, pos[idim]*lunit + field.relem(i,j,k) );
                }
            }
        }
    }
}

template<typename field_t>
void set_velocities_old( container& particles, lattice lattice_type, int idim, field_t& field )
{
    const size_t num_p_in_load = field.local_size();

    for( size_t i=0,ipcount=0; i<field.size(0); ++i ){
        for( size_t j=0; j<field.size(1); ++j){
            for( size_t k=0; k<field.size(2); ++k){
                particles.set_vel( ipcount++, idim, field.relem(i,j,k) );
            }
        }
    }

    if( lattice_type == particle::lattice_bcc ){
        field.shift_field( 0.5, 0.5, 0.5 );
        auto ipcount0 = num_p_in_load;
        for( size_t i=0,ipcount=ipcount0; i<field.size(0); ++i ){
            for( size_t j=0; j<field.size(1); ++j){
                for( size_t k=0; k<field.size(2); ++k){
                    particles.set_vel( ipcount++, idim, field.relem(i,j,k) );
                }
            }
        }
    }
    else if( lattice_type == particle::lattice_fcc ){ 
        // 0.5 0.5 0.0
        field.shift_field( 0.5, 0.5, 0.0 );
        auto ipcount0 = num_p_in_load;
        for( size_t i=0,ipcount=ipcount0; i<field.size(0); ++i ){
            for( size_t j=0; j<field.size(1); ++j){
                for( size_t k=0; k<field.size(2); ++k){
                    particles.set_vel( ipcount++, idim, field.relem(i,j,k) );
                }
            }
        }
        // 0.0 0.5 0.5
        field.shift_field( -0.5, 0.0, 0.5 );
        ipcount0 = 2*num_p_in_load;
        for( size_t i=0,ipcount=ipcount0; i<field.size(0); ++i ){
            for( size_t j=0; j<field.size(1); ++j){
                for( size_t k=0; k<field.size(2); ++k){
                    particles.set_vel( ipcount++, idim, field.relem(i,j,k) );
                }
            }
        }
        // 0.5 0.0 0.5
        field.shift_field( 0.5, -0.5, 0.0 );
        ipcount0 = 3*num_p_in_load;
        for( size_t i=0,ipcount=ipcount0; i<field.size(0); ++i ){
            for( size_t j=0; j<field.size(1); ++j){
                for( size_t k=0; k<field.size(2); ++k){
                    particles.set_vel( ipcount++, idim, field.relem(i,j,k) );
                }
            }
        }
    }
}


} // end namespace particles
