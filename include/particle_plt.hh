#pragma once

#include <general.hh>
#include <unistd.h> // for unlink

#include <iostream>
#include <fstream>

#include <random>

#include <particle_generator.hh>
#include <grid_fft.hh>
#include <mat3.hh>

// #define PRODUCTION

namespace particle{
//! implement Marcos et al. PLT calculation

class lattice_gradient{
private:
    const real_t boxlen_;
    const size_t ngmapto_, ngrid_, ngrid32_;
    const real_t mapratio_;
    Grid_FFT<real_t,false> D_xx_, D_xy_, D_xz_, D_yy_, D_yz_, D_zz_;
    Grid_FFT<real_t,false> grad_x_, grad_y_, grad_z_;
    std::vector<vec3<real_t>> vectk_;
    std::vector<vec3<int>> ico_, vecitk_;

    void init_D()
    {
        constexpr real_t pi = M_PI;
        constexpr real_t twopi = 2.0*M_PI;
        constexpr real_t fourpi = 4.0*M_PI;
        constexpr real_t sqrtpi = std::sqrt(M_PI);
        constexpr real_t pi32   = std::pow(M_PI,1.5);

        const int charge_multiplicity = 2;

        //! === vectors, reciprocals and normals for the SC lattice ===
        const int charge_fac_sc = 1;
        const mat3<real_t> mat_bravais_sc{
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 
        };
        const mat3<real_t> mat_reciprocal_sc{
            twopi, 0.0, 0.0,
            0.0, twopi, 0.0,
            0.0, 0.0, twopi,
        };
        const std::vector<vec3<real_t>> normals_sc{
            {pi,0.,0.},{-pi,0.,0.},
            {0.,pi,0.},{0.,-pi,0.},
            {0.,0.,pi},{0.,0.,-pi},
        };
        

        //! === vectors, reciprocals and normals for the BCC lattice ===
        const int charge_fac_bcc = 2;
        const mat3<real_t> mat_bravais_bcc{
            1.0, 0.0, 0.5,
            0.0, 1.0, 0.5,
            0.0, 0.0, 0.5, 
        };
        const mat3<real_t> mat_reciprocal_bcc{
            twopi, 0.0, 0.0,
            0.0, twopi, 0.0,
            -twopi, -twopi, fourpi,
        };
        const std::vector<vec3<real_t>> normals_bcc{
            {0.,pi,pi},{0.,-pi,pi},{0.,pi,-pi},{0.,-pi,-pi},
            {pi,0.,pi},{-pi,0.,pi},{pi,0.,-pi},{-pi,0.,-pi},
            {pi,pi,0.},{-pi,pi,0.},{pi,-pi,0.},{-pi,-pi,0.}
        };
        

        //! === vectors, reciprocals and normals for the FCC lattice ===
        const int charge_fac_fcc = 4;
        const mat3<real_t> mat_bravais_fcc{
            0.0, 0.5, 0.0,
            0.5, 0.0, 1.0,
            0.5, 0.5, 0.0, 
        };
        const mat3<real_t> mat_reciprocal_fcc{
            -fourpi, fourpi, twopi,
            0.0, 0.0, twopi,
            fourpi, 0.0, -twopi,
        };
        const std::vector<vec3<real_t>> normals_fcc{
            {twopi,0.,0.},{-twopi,0.,0.},
            {0.,twopi,0.},{0.,-twopi,0.},
            {0.,0.,twopi},{0.,0.,-twopi},
            {+pi,+pi,+pi},{+pi,+pi,-pi},
            {+pi,-pi,+pi},{+pi,-pi,-pi},
            {-pi,+pi,+pi},{-pi,+pi,-pi},
            {-pi,-pi,+pi},{-pi,-pi,-pi},
        };
        
        //! select the properties for the chosen lattice
        const int ilat = 2; // 0 = sc, 1 = bcc, 2 = fcc

        const auto mat_bravais    = (ilat==2)? mat_bravais_fcc : (ilat==1)? mat_bravais_bcc : mat_bravais_sc;
        const auto mat_reciprocal = (ilat==2)? mat_reciprocal_fcc : (ilat==1)? mat_reciprocal_bcc : mat_reciprocal_sc;
        const auto normals        = (ilat==2)? normals_fcc : (ilat==1)? normals_bcc : normals_sc;
        const auto charge_fac     = (ilat==2)? charge_fac_fcc : (ilat==1)? charge_fac_bcc : charge_fac_sc;

        const size_t nlattice = ngrid_;//16;
        const real_t dx = 1.0/real_t(nlattice);

        const real_t eta = 4.0;//nlattice;//4.0; //2.0/ngrid_; // Ewald cutoff shall be 2 cells
        const real_t alpha = 1.0/std::sqrt(2)/eta;
        const real_t alpha2 = alpha*alpha;
        const real_t alpha3 = alpha2*alpha;
        
        const real_t charge = 1.0/std::pow(real_t(nlattice),3)/charge_fac;
        const real_t fft_norm   = 1.0/std::pow(real_t(nlattice),3.0);
        const real_t fft_norm12 = 1.0/std::pow(real_t(nlattice),1.5);

        //! just a Kronecker \delta_ij
        auto kronecker = []( int i, int j ) -> real_t { return (i==j)? 1.0 : 0.0; };

        auto add_greensftide_sr = [&]( mat3<real_t>& D, const vec3<real_t>& d ) -> void {
            auto r = d.norm();
            if( r< 1e-14 ) return; // return zero for r=0

            const real_t r2(r*r), r3(r2*r), r5(r3*r2);
            const real_t K1( -alpha3/pi32 * std::exp(-alpha2*r2)/r2 );
            const real_t K2( (std::erfc(alpha*r) + 2.0*alpha/sqrtpi*std::exp(-alpha2*r2)*r)/fourpi );
            
            for( int mu=0; mu<3; ++mu ){
                for( int nu=mu; nu<3; ++nu ){
                    real_t dd( d[mu]*d[nu] * K1 + (kronecker(mu,nu)/r3 - 3.0 * (d[mu]*d[nu])/r5) * K2 );
                    D(mu,nu) += dd;
                    D(nu,mu) += (mu!=nu)? dd : 0.0;
                }
            }
        };

        auto add_greensftide_lr = [&]( mat3<real_t>& D, const vec3<real_t>& k, const vec3<real_t>& r ) -> void {
            real_t kmod2 = k.norm_squared();
            real_t term = std::exp(-kmod2/(4*alpha2))*std::cos(k.dot(r)) / kmod2 * fft_norm;
            for( int mu=0; mu<3; ++mu ){
                for( int nu=mu; nu<3; ++nu ){
                    auto dd = k[mu] * k[nu] * term;
                    D(mu,nu) += dd;
                    D(nu,mu) += (mu!=nu)? dd : 0.0;
                }
            }
        };
        
        constexpr ptrdiff_t lnumber = 4, knumber = 4;
        const int numb = 1;

        vectk_.assign(D_xx_.memsize(),vec3<real_t>());
        ico_.assign(D_xx_.memsize(),vec3<int>());
        vecitk_.assign(D_xx_.memsize(),vec3<int>());

        #pragma omp parallel 
        {
            //... temporary to hold values of the dynamical matrix 
            mat3<real_t> matD(0.0);

            #pragma omp for
            for( size_t i=0; i<nlattice; ++i ){
                for( size_t j=0; j<nlattice; ++j ){
                    for( size_t k=0; k<nlattice; ++k ){
                        // compute lattice site vector from (i,j,k) multiplying Bravais base matrix, and wrap back to box
                        const vec3<real_t> x_ijk({dx*real_t(i),dx*real_t(j),dx*real_t(k)});
                        const vec3<real_t> ar = (mat_bravais * x_ijk).wrap_abs();

                        //... zero temporary matrix
                        matD.zero();        

                        // add real-space part of dynamical matrix, periodic copies
                        for( ptrdiff_t ix=-lnumber; ix<=lnumber; ix++ ){
                            for( ptrdiff_t iy=-lnumber; iy<=lnumber; iy++ ){
                                for( ptrdiff_t iz=-lnumber; iz<=lnumber; iz++ ){      
                                    const vec3<real_t> n_ijk({real_t(ix),real_t(iy),real_t(iz)});            
                                    const vec3<real_t> dr(ar - mat_bravais * n_ijk);
                                    add_greensftide_sr(matD, dr);
                                }
                            }
                        }

                        // add k-space part of dynamical matrix
                        for( ptrdiff_t ix=-knumber; ix<=knumber; ix++ ){
                            for( ptrdiff_t iy=-knumber; iy<=knumber; iy++ ){
                                for( ptrdiff_t iz=-knumber; iz<=knumber; iz++ ){                      
                                    if(std::abs(ix)+std::abs(iy)+std::abs(iz) != 0){
                                        const vec3<real_t> k_ijk({real_t(ix)/nlattice,real_t(iy)/nlattice,real_t(iz)/nlattice});
                                        const vec3<real_t> ak( mat_reciprocal * k_ijk);

                                        add_greensftide_lr(matD, ak, ar );
                                    }
                                }
                            }
                        } 

                        D_xx_.relem(i,j,k) = matD(0,0) * charge;
                        D_xy_.relem(i,j,k) = matD(0,1) * charge;
                        D_xz_.relem(i,j,k) = matD(0,2) * charge;
                        D_yy_.relem(i,j,k) = matD(1,1) * charge;
                        D_yz_.relem(i,j,k) = matD(1,2) * charge;
                        D_zz_.relem(i,j,k) = matD(2,2) * charge;
                    }
                }
            }
        } // end omp parallel region

        // fix r=0 with background density (added later in Fourier space)
        D_xx_.relem(0,0,0) = 1.0/3.0;
        D_xy_.relem(0,0,0) = 0.0;
        D_xz_.relem(0,0,0) = 0.0;
        D_yy_.relem(0,0,0) = 1.0/3.0;
        D_yz_.relem(0,0,0) = 0.0;
        D_zz_.relem(0,0,0) = 1.0/3.0;

        D_xx_.FourierTransformForward();
        D_xy_.FourierTransformForward();
        D_xz_.FourierTransformForward();
        D_yy_.FourierTransformForward();
        D_yz_.FourierTransformForward();
        D_zz_.FourierTransformForward();

        if (CONFIG::MPI_task_rank == 0)
            unlink("debug.hdf5");
        D_xx_.Write_to_HDF5("debug.hdf5","Dxx");
        D_xy_.Write_to_HDF5("debug.hdf5","Dxy");
        D_xz_.Write_to_HDF5("debug.hdf5","Dxz");
        D_yy_.Write_to_HDF5("debug.hdf5","Dyy");
        D_yz_.Write_to_HDF5("debug.hdf5","Dyz");
        D_zz_.Write_to_HDF5("debug.hdf5","Dzz");

        std::ofstream ofs2("test_brillouin.txt");
        
        #pragma omp parallel
        {
            // thread private matrix representation
            mat3<real_t> D;
            vec3<real_t> eval, evec1, evec2, evec3;

            #pragma omp for
            for( size_t i=0; i<D_xx_.size(0); i++ )
            {
                for( size_t j=0; j<D_xx_.size(1); j++ )
                {
                    for( size_t k=0; k<D_xx_.size(2); k++ )
                    {
                        vec3<real_t> kv = D_xx_.get_k<real_t>(i,j,k);
                        const real_t kmod  = kv.norm()/mapratio_/boxlen_;

                        // put matrix elements into actual matrix
                        D(0,0) = std::real(D_xx_.kelem(i,j,k)) / fft_norm12;
                        D(0,1) = D(1,0) = std::real(D_xy_.kelem(i,j,k)) / fft_norm12;
                        D(0,2) = D(2,0) = std::real(D_xz_.kelem(i,j,k)) / fft_norm12;
                        D(1,1) = std::real(D_yy_.kelem(i,j,k)) / fft_norm12;
                        D(1,2) = D(2,1) = std::real(D_yz_.kelem(i,j,k)) / fft_norm12;
                        D(2,2) = std::real(D_zz_.kelem(i,j,k)) / fft_norm12;

                        // compute eigenstructure of matrix
                        D.eigen(eval, evec1, evec2, evec3);

                        D_xx_.kelem(i,j,k) = eval[2];
                        D_yy_.kelem(i,j,k) = eval[1];
                        D_zz_.kelem(i,j,k) = eval[0];

                        D_xy_.kelem(i,j,k) = evec3[0];
                        D_xz_.kelem(i,j,k) = evec3[1];
                        D_yz_.kelem(i,j,k) = evec3[2];


                        vec3<real_t> a({0.,0.,0.});

                        auto idx = D_xx_.get_idx(i,j,k);

                        vec3<real_t> ar = D_xx_.get_k<real_t>(i,j,k) / (twopi*ngrid_);
                        a = mat_reciprocal * ar;

                        // translate the k-vectors into the "candidate" FBZ
                        vec3<real_t> anum;
                        for( int l1=-numb; l1<=numb; ++l1 ){
                            anum[0] = real_t(l1);
                            for( int l2=-numb; l2<=numb; ++l2 ){
                                anum[1] = real_t(l2);
                                for( int l3=-numb; l3<=numb; ++l3 ){
                                    anum[2] = real_t(l3);

                                    // vectk_[idx] = a;
                                    vectk_[idx] = a + mat_reciprocal * anum;

                                    // for( int l=0; l<3; l++ ){
                                    //     for( int m=0; m<3; m++){
                                    //         // project k on reciprocal basis
                                    //         vectk_[idx][l] += anum[m]*bcc_reciprocal[m][l];
                                    //     }
                                    // }
                                    // check if in first Brillouin zone
                                    bool btest=true;
                                    for( size_t l=0; l<normals.size(); ++l ){
                                        real_t amod2 = 0.0;
                                        real_t scalar = 0.0;
                                        for( int m=0; m<3; m++ ){
                                            amod2  += normals[l][m]*normals[l][m];
                                            scalar += normals[l][m]*vectk_[idx][m];
                                        }
                                        //real_t amod = std::sqrt(amod2);
                                        //if( scalar/amod > amod*1.0001 ){ btest=false; break; }
                                        if( scalar > 1.01 * amod2 ){ btest=false; break; }
                                    }
                                    if( btest ){
                                        
                                        vecitk_[idx][0] = std::round(vectk_[idx][0]*(ngrid_)/twopi);
                                        vecitk_[idx][1] = std::round(vectk_[idx][1]*(ngrid_)/twopi);
                                        vecitk_[idx][2] = std::round(vectk_[idx][2]*(ngrid_)/twopi);

                                        ico_[idx][0] = std::round((ar[0]+l1) * ngrid_);
                                        ico_[idx][1] = std::round((ar[1]+l2) * ngrid_);
                                        ico_[idx][2] = std::round((ar[2]+l3) * ngrid_);

                                        ofs2 << vectk_[idx].norm() << " " << kv.norm() << " " << std::real(D_xx_.kelem(i,j,k)) << " " << std::real(D_yy_.kelem(i,j,k)) << " " << std::real(D_zz_.kelem(i,j,k)) << std::endl;
                                        // ofs2 << kv.x/twopi << ", " << kv.y/twopi << ", " << kv.z/twopi << ", " << vecitk_[idx].x << ", " << vecitk_[idx].y << ", " << vecitk_[idx].z << ", " << ico_[idx][0] << ", " << ico_[idx][1] << ", " << ico_[idx][2] << std::endl;
                                        // ofs2 << kv.x/twopi << ", " << kv.y/twopi << ", " << kv.z/twopi << ", " << -vecitk_[idx].x << ", " << -vecitk_[idx].y << ", " << -vecitk_[idx].z << ", " << ico_[idx][0] << ", " << ico_[idx][1] << ", " << ico_[idx][2] << std::endl;
                                        
                                        goto endloop;
                                    }
                                }
                            }
                        }                    endloop: ;
                    }
                }
            }
        }

        

        D_xx_.Write_to_HDF5("debug.hdf5","mu1");
        D_xy_.Write_to_HDF5("debug.hdf5","mu2");
        D_xz_.Write_to_HDF5("debug.hdf5","mu3");
        D_yy_.Write_to_HDF5("debug.hdf5","e1x");
        D_yz_.Write_to_HDF5("debug.hdf5","e1y");
        D_zz_.Write_to_HDF5("debug.hdf5","e1z");
    }

    void init_D__old()
    {
        constexpr real_t pi = M_PI, twopi = 2.0*M_PI;

        const std::vector<vec3<real_t>> normals_bcc{
            {0.,pi,pi},{0.,-pi,pi},{0.,pi,-pi},{0.,-pi,-pi},
            {pi,0.,pi},{-pi,0.,pi},{pi,0.,-pi},{-pi,0.,-pi},
            {pi,pi,0.},{-pi,pi,0.},{pi,-pi,0.},{-pi,-pi,0.}
        };

        const std::vector<vec3<real_t>> bcc_reciprocal{
            {twopi,0.,-twopi}, {0.,twopi,-twopi}, {0.,0.,2*twopi}
        };

        const real_t eta = 2.0/ngrid_; // Ewald cutoff shall be 2 cells
        const real_t alpha = 1.0/std::sqrt(2)/eta;
        const real_t alpha2 = alpha*alpha;
        const real_t alpha3 = alpha2*alpha;
        const real_t sqrtpi = std::sqrt(M_PI);
        const real_t pi32   = std::pow(M_PI,1.5);

        //! just a Kronecker \delta_ij
        auto kronecker = []( int i, int j ) -> real_t { return (i==j)? 1.0 : 0.0; };

        //! short range component of Ewald sum, eq. (A2) of Marcos (2008)
        auto greensftide_sr = [&]( int mu, int nu, const vec3<real_t>& vR, const vec3<real_t>& vP ) -> real_t {
            auto d = vR-vP;
            auto r = d.norm();
            if( r< 1e-14 ) return 0.0; // let's return nonsense for r=0, and fix it later!
            real_t val{0.0};
            val -= d[mu]*d[nu]/(r*r) * alpha3/pi32 * std::exp(-alpha*alpha*r*r);
            val += 1.0/(4.0*M_PI)*(kronecker(mu,nu)/std::pow(r,3) - 3.0 * (d[mu]*d[nu])/std::pow(r,5)) * 
                (std::erfc(alpha*r) + 2.0*alpha/sqrtpi*std::exp(-alpha*alpha*r*r)*r);
            return val;
        };

        //! sums mirrored copies of short-range component of Ewald sum
        auto evaluate_D = [&]( int mu, int nu, const vec3<real_t>& v ) -> real_t{
            real_t sr = 0.0;
            constexpr int N = 3; // number of repeated copies ±N per dimension
            int count = 0;
            for( int i=-N; i<=N; ++i ){
                for( int j=-N; j<=N; ++j ){
                    for( int k=-N; k<=N; ++k ){
                        if( std::abs(i)+std::abs(j)+std::abs(k) <= N ){
                            //sr += greensftide_sr( mu, nu, v, {real_t(i),real_t(j),real_t(k)} );
                            sr += greensftide_sr( mu, nu, v, {real_t(i),real_t(j),real_t(k)} );
                            sr += greensftide_sr( mu, nu, v, {real_t(i)+0.5,real_t(j)+0.5,real_t(k)+0.5} );
                            count += 2;

                            // sr += greensftide_sr( mu, nu, v, {real_t(i)+0.5,real_t(j)+0.5,real_t(k)+0.5} )/16;
                            // sr += greensftide_sr( mu, nu, v, {real_t(i)+0.5,real_t(j)+0.5,real_t(k)-0.5} )/16;
                            // sr += greensftide_sr( mu, nu, v, {real_t(i)+0.5,real_t(j)-0.5,real_t(k)+0.5} )/16;
                            // sr += greensftide_sr( mu, nu, v, {real_t(i)+0.5,real_t(j)-0.5,real_t(k)-0.5} )/16;
                            // sr += greensftide_sr( mu, nu, v, {real_t(i)-0.5,real_t(j)+0.5,real_t(k)+0.5} )/16;
                            // sr += greensftide_sr( mu, nu, v, {real_t(i)-0.5,real_t(j)+0.5,real_t(k)-0.5} )/16;
                            // sr += greensftide_sr( mu, nu, v, {real_t(i)-0.5,real_t(j)-0.5,real_t(k)+0.5} )/16;
                            // sr += greensftide_sr( mu, nu, v, {real_t(i)-0.5,real_t(j)-0.5,real_t(k)-0.5} )/16;
                        }
                    }
                }
            }
            return sr / count;
        };

        //! fill D_ij array with short range evaluated function
        #pragma omp parallel for
        for( size_t i=0; i<ngrid_; i++ ){
            vec3<real_t>  p;
            p.x = real_t(i)/ngrid_;
            for( size_t j=0; j<ngrid_; j++ ){
                p.y = real_t(j)/ngrid_;
                for( size_t k=0; k<ngrid_; k++ ){
                    p.z = real_t(k)/ngrid_;
                    D_xx_.relem(i,j,k) = evaluate_D(0,0,p);
                    D_xy_.relem(i,j,k) = evaluate_D(0,1,p);
                    D_xz_.relem(i,j,k) = evaluate_D(0,2,p);
                    D_yy_.relem(i,j,k) = evaluate_D(1,1,p);
                    D_yz_.relem(i,j,k) = evaluate_D(1,2,p);
                    D_zz_.relem(i,j,k) = evaluate_D(2,2,p);
                }   
            }    
        }
        // fix r=0 with background density (added later in Fourier space)
        D_xx_.relem(0,0,0) = 0.0;
        D_xy_.relem(0,0,0) = 0.0;
        D_xz_.relem(0,0,0) = 0.0;
        D_yy_.relem(0,0,0) = 0.0;
        D_yz_.relem(0,0,0) = 0.0;
        D_zz_.relem(0,0,0) = 0.0;
        

        // Fourier transform all six components
        D_xx_.FourierTransformForward();
        D_xy_.FourierTransformForward();
        D_xz_.FourierTransformForward();
        D_yy_.FourierTransformForward();
        D_yz_.FourierTransformForward();
        D_zz_.FourierTransformForward();

        const real_t rho0 = std::pow(real_t(ngrid_),1.5); //mass of one particle in Fourier space
        const real_t nfac = 1.0/std::pow(real_t(ngrid_),1.5);

        #pragma omp parallel
        {
            // thread private matrix representation
            mat3<real_t> D;
            vec3<real_t> eval, evec1, evec2, evec3;
        
            #pragma omp for
            for( size_t i=0; i<D_xx_.size(0); i++ )
            {
                for( size_t j=0; j<D_xx_.size(1); j++ )
                {
                    for( size_t k=0; k<D_xx_.size(2); k++ )
                    {
                        vec3<real_t> kv = D_xx_.get_k<real_t>(i,j,k);
                        auto& b=bcc_reciprocal;
                        vec3<real_t> kvc = { b[0][0]*kvc[0]+b[1][0]*kvc[1]+b[2][0]*kvc[2],
                                            b[0][1]*kvc[0]+b[1][1]*kvc[1]+b[2][1]*kvc[2],
                                            b[0][2]*kvc[0]+b[1][2]*kvc[1]+b[2][2]*kvc[2] };
                        // vec3<real_t> kv = {kvc.dot(bcc_reciprocal[0]),kvc.dot(bcc_reciprocal[1]),kvc.dot(bcc_reciprocal[2])};
                        const real_t kmod2 = kv.norm_squared();

                        // long range component of Ewald sum
                        //ccomplex_t shift = 1.0;//std::exp(ccomplex_t(0.0,0.5*(kv[0] + kv[1] + kv[2])* D_xx_.get_dx()[0]));
                        ccomplex_t phi0 = -rho0 * std::exp(-0.5*eta*eta*kmod2) / kmod2;
                        phi0 = (phi0==phi0)? phi0 : 0.0; // catch NaN from division by zero when kmod2=0


                        const int nn = 3;
                        size_t nsum = 0;
                        ccomplex_t ff = 0.0;
                        for( int is=-nn;is<=nn;is++){
                            for( int js=-nn;js<=nn;js++){
                                for( int ks=-nn;ks<=nn;ks++){
                                    if( std::abs(is)+std::abs(js)+std::abs(ks) <= nn ){
                                        ff += std::exp(ccomplex_t(0.0,(((is)*kv[0] + (js)*kv[1] + (ks)*kv[2]))));
                                        ff += std::exp(ccomplex_t(0.0,(((0.5+is)*kv[0] + (0.5+js)*kv[1] + (0.5+ks)*kv[2]))));
                                        ++nsum;
                                    }
                                }
                            }    
                        }
                        ff /= nsum;
                        // ccomplex_t ff = 1.0; 
                        // ccomplex_t ff = (0.5+0.5*std::exp(ccomplex_t(0.0,0.5*(kv[0] + kv[1] + kv[2]))));
                        // assemble short-range + long_range of Ewald sum and add DC component to trace
                        D_xx_.kelem(i,j,k) = ff*((D_xx_.kelem(i,j,k) - kv[0]*kv[0] * phi0)*nfac) + 1.0/3.0;
                        D_xy_.kelem(i,j,k) = ff*((D_xy_.kelem(i,j,k) - kv[0]*kv[1] * phi0)*nfac);
                        D_xz_.kelem(i,j,k) = ff*((D_xz_.kelem(i,j,k) - kv[0]*kv[2] * phi0)*nfac);
                        D_yy_.kelem(i,j,k) = ff*((D_yy_.kelem(i,j,k) - kv[1]*kv[1] * phi0)*nfac) + 1.0/3.0;
                        D_yz_.kelem(i,j,k) = ff*((D_yz_.kelem(i,j,k) - kv[1]*kv[2] * phi0)*nfac);
                        D_zz_.kelem(i,j,k) = ff*((D_zz_.kelem(i,j,k) - kv[2]*kv[2] * phi0)*nfac) + 1.0/3.0;

                    }
                }
            }

            D_xx_.kelem(0,0,0) = 1.0/3.0;
            D_xy_.kelem(0,0,0) = 0.0;
            D_xz_.kelem(0,0,0) = 0.0;
            D_yy_.kelem(0,0,0) = 1.0/3.0;
            D_yz_.kelem(0,0,0) = 0.0;
            D_zz_.kelem(0,0,0) = 1.0/3.0;

            #pragma omp for
            for( size_t i=0; i<D_xx_.size(0); i++ )
            {
                for( size_t j=0; j<D_xx_.size(1); j++ )
                {
                    for( size_t k=0; k<D_xx_.size(2); k++ )
                    {
                        vec3<real_t> kv = D_xx_.get_k<real_t>(i,j,k);
                        const real_t kmod  = kv.norm()/mapratio_/boxlen_;

                        // put matrix elements into actual matrix
                        D = { std::real(D_xx_.kelem(i,j,k)), std::real(D_xy_.kelem(i,j,k)), std::real(D_xz_.kelem(i,j,k)),
                              std::real(D_yy_.kelem(i,j,k)), std::real(D_yz_.kelem(i,j,k)), std::real(D_zz_.kelem(i,j,k)) };
                        
                        // compute eigenstructure of matrix
                        D.eigen(eval, evec1, evec2, evec3);

                        // store in diagonal components of D_ij
                        D_xx_.kelem(i,j,k) =  ccomplex_t(0.0,kmod) * evec3.x;
                        D_yy_.kelem(i,j,k) =  ccomplex_t(0.0,kmod) * evec3.y;
                        D_zz_.kelem(i,j,k) =  ccomplex_t(0.0,kmod) * evec3.z;

                        auto norm = (kv.norm()/kv.dot(evec3));
                        if ( std::abs(kv.dot(evec3)) < 1e-10 || kv.norm() < 1e-10 ) norm = 0.0; 
#ifdef PRODUCTION
                        D_xx_.kelem(i,j,k) *= norm;
                        D_yy_.kelem(i,j,k) *= norm;
                        D_zz_.kelem(i,j,k) *= norm;

                        // spatially dependent correction to vfact = \dot{D_+}/D_+
                        D_xy_.kelem(i,j,k) = 1.0/(0.25*(std::sqrt(1.+24*eval[2])-1.));
#else

                        D_xx_.kelem(i,j,k) = eval[2];
                        D_yy_.kelem(i,j,k) = eval[1];
                        D_zz_.kelem(i,j,k) = eval[0];

                        D_xy_.kelem(i,j,k) = evec3[0];
                        D_xz_.kelem(i,j,k) = evec3[1];
                        D_yz_.kelem(i,j,k) = evec3[2];
#endif
                    }
                }
            }
        }
#ifdef PRODUCTION
        D_xy_.kelem(0,0,0) = 1.0;
#endif

        //////////////////////////////////////////
        std::string filename("plt_test.hdf5");
        unlink(filename.c_str());
    #if defined(USE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
    #endif
    //     rho.Write_to_HDF5(filename, "rho");
        D_xx_.Write_to_HDF5(filename, "omega1");
        D_yy_.Write_to_HDF5(filename, "omega2");
        D_zz_.Write_to_HDF5(filename, "omega3");
        D_xy_.Write_to_HDF5(filename, "e1_x");
        D_xz_.Write_to_HDF5(filename, "e1_y");
        D_yz_.Write_to_HDF5(filename, "e1_z");

    }


public:
    // real_t boxlen, size_t ngridother
    explicit lattice_gradient( ConfigFile& the_config, size_t ngridself=32 )
    : boxlen_( the_config.GetValue<double>("setup", "BoxLength") ), 
      ngmapto_( the_config.GetValue<size_t>("setup", "GridRes") ), 
      ngrid_( ngridself ), ngrid32_( std::pow(ngrid_, 1.5) ), mapratio_(real_t(ngrid_)/real_t(ngmapto_)),
      D_xx_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}), D_xy_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}),
      D_xz_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}), D_yy_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}),
      D_yz_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}), D_zz_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}),
      grad_x_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}), grad_y_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}),
      grad_z_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0})
    { 
        csoca::ilog << "-------------------------------------------------------------------------------" << std::endl;
        std::string lattice_str = the_config.GetValueSafe<std::string>("setup","ParticleLoad","sc");
        const lattice lattice_type = 
            ((lattice_str=="bcc")? lattice_bcc 
            : ((lattice_str=="fcc")? lattice_fcc 
            : ((lattice_str=="rsc")? lattice_rsc 
            : lattice_sc)));

        if( lattice_type != lattice_sc){
            csoca::elog << "PLT not implemented for chosen lattice type! Currently only SC." << std::endl;
            abort();
        }

        csoca::ilog << "PLT corrections for SC lattice will be computed on " << ngrid_ << "**3 mesh" << std::endl;

// #if defined(USE_MPI)
//         if( CONFIG::MPI_task_size>1 )
//         {
//             csoca::elog << "PLT not implemented for MPI, cannot run with more than 1 task currently!" << std::endl;
//             abort();
//         }
// #endif 

        double wtime = get_wtime();
        csoca::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing PLT eigenmodes "<< std::flush;
        
        init_D();

        csoca::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime()-wtime << "s" << std::endl;
    }

    inline ccomplex_t gradient( const int idim, std::array<size_t,3> ijk ) const
    {
        real_t ix = ijk[0]*mapratio_, iy = ijk[1]*mapratio_, iz = ijk[2]*mapratio_;
        if( idim == 0 )    return D_xx_.get_cic_kspace({ix,iy,iz});
        else if( idim == 1 ) return D_yy_.get_cic_kspace({ix,iy,iz});
        return D_zz_.get_cic_kspace({ix,iy,iz});
    }

    inline ccomplex_t vfac_corr( std::array<size_t,3> ijk ) const
    {
        real_t ix = ijk[0]*mapratio_, iy = ijk[1]*mapratio_, iz = ijk[2]*mapratio_;
        return D_xy_.get_cic_kspace({ix,iy,iz});
    }

};

}