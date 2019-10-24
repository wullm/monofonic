#pragma once

namespace particle {

enum lattice{
    lattice_sc=0, lattice_bcc=1, lattice_fcc=2
};

template<typename field_t>
void initialize_lattice( container& particles, lattice lattice_type, const field_t& field ){
    const size_t num_p_in_load = field.local_size();
    const size_t overload = 1<<lattice_type; // 1 for sc, 2 for bcc, 4 for fcc

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
void set_positions( container& particles, lattice lattice_type, int idim, real_t lunit, field_t& field )
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
void set_velocities( container& particles, lattice lattice_type, int idim, field_t& field )
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
