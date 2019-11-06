#pragma once

#include <general.hh>
#include <unistd.h> // for unlink

#include <iostream>
#include <fstream>

#include <random>

#include <grid_fft.hh>
#include <mat3.hh>

namespace particle{
//! implement Marcos et al. PLT calculation

class lattice_gradient{
private:
    const real_t boxlen_;
    const size_t ngmapto_, ngrid_, ngrid32_;
    const real_t mapratio_;
    Grid_FFT<real_t> D_xx_, D_xy_, D_xz_, D_yy_, D_yz_, D_zz_;
    Grid_FFT<real_t> grad_x_, grad_y_, grad_z_;

    void init_D()
    {
        const real_t eta = 2.0/ngrid_; // Ewald cutoff shall be 2 cells
        const real_t alpha = 1.0/std::sqrt(2)/eta;
        const real_t alpha2 = alpha*alpha;
        const real_t alpha3 = alpha2*alpha;
        const real_t sqrtpi = std::sqrt(M_PI);
        const real_t pi32   = std::pow(M_PI,1.5);

        //! just a Kronecker \delta_ij
        auto kronecker = []( int i, int j ) -> real_t { return (i==j)? 1.0 : 0.0; };

        //! just a sign function
        auto sign = []( real_t x ) -> real_t { return (x<0.0)? -1.0 : 1.0; };

        //! short range component of Ewald sum, eq. (A2) of Marcos (2008)
        auto greensftide_sr = [&]( int mu, int nu, const vec3<real_t>& vR, const vec3<real_t>& vP ) -> real_t {
            auto d = vR-vP;
            auto r = d.norm();
            // if( r< 1e-14 ) return 0.0; // let's return nonsense for r=0, and fix it later!
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
            for( int i=-N; i<=N; ++i ){
                for( int j=-N; j<=N; ++j ){
                    for( int k=-N; k<=N; ++k ){
                        if( std::abs(i)+std::abs(j)+std::abs(k) <= N ){
                            sr += greensftide_sr( mu, nu, v, {real_t(i),real_t(j),real_t(k)} );
                        }
                    }
                }
            }
            return sr;
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
            mat3s<real_t> D;
            vec3<real_t> eval, evec1, evec2, evec3;
        
            #pragma omp for
            for( size_t i=0; i<D_xx_.size(0); i++ )
            {
                for( size_t j=0; j<D_xx_.size(1); j++ )
                {
                    for( size_t k=0; k<D_xx_.size(2); k++ )
                    {
                        vec3<real_t> kv = D_xx_.get_k<real_t>(i,j,k);
                        const real_t kmod2 = kv.norm_squared();

                        // long range component of Ewald sum
                        real_t phi0 = -rho0 * std::exp(-0.5*eta*eta*kmod2) / kmod2;
                        phi0 = (phi0==phi0)? phi0 : 0.0; // catch NaN from division by zero when kmod2=0

                        // assemble short-range + long_range of Ewald sum and add DC component to trace
                        D_xx_.kelem(i,j,k) = (D_xx_.kelem(i,j,k) - kv[0]*kv[0] * phi0)*nfac + 1.0/3.0;
                        D_xy_.kelem(i,j,k) = (D_xy_.kelem(i,j,k) - kv[0]*kv[1] * phi0)*nfac;
                        D_xz_.kelem(i,j,k) = (D_xz_.kelem(i,j,k) - kv[0]*kv[2] * phi0)*nfac;
                        D_yy_.kelem(i,j,k) = (D_yy_.kelem(i,j,k) - kv[1]*kv[1] * phi0)*nfac + 1.0/3.0;
                        D_yz_.kelem(i,j,k) = (D_yz_.kelem(i,j,k) - kv[1]*kv[2] * phi0)*nfac;
                        D_zz_.kelem(i,j,k) = (D_zz_.kelem(i,j,k) - kv[2]*kv[2] * phi0)*nfac + 1.0/3.0;

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

                        D_xx_.kelem(i,j,k) *= norm;
                        D_yy_.kelem(i,j,k) *= norm;
                        D_zz_.kelem(i,j,k) *= norm;

                        // spatially dependent correction to vfact = \dot{D_+}/D_+
                        D_xy_.kelem(i,j,k) = 1.0/(0.25*(std::sqrt(1.+24*eval[2])-1.));
                    }
                }
            }
        }

        D_xy_.kelem(0,0,0) = 1.0;

    }

public:
    explicit lattice_gradient( real_t boxlen, size_t ngridother, size_t ngridself=64 )
    : boxlen_(boxlen), ngmapto_(ngridother), ngrid_( ngridself ), ngrid32_( std::pow(ngrid_, 1.5) ), mapratio_(real_t(ngrid_)/real_t(ngmapto_)),
      D_xx_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}), D_xy_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}),
      D_xz_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}), D_yy_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}),
      D_yz_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}), D_zz_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}),
      grad_x_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}), grad_y_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0}),
      grad_z_({ngrid_, ngrid_, ngrid_}, {1.0,1.0,1.0})
    { 
        csoca::ilog << "-------------------------------------------------------------------------------" << std::endl;
        double wtime = get_wtime();
        csoca::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing PLT lattice eigenmodes "<< std::flush;
        
        init_D();

        csoca::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime()-wtime << "s" << std::endl;
    }

    inline ccomplex_t gradient( const int idim, std::array<size_t,3> ijk ) const
    {
        real_t ix = ijk[0]*mapratio_, iy = ijk[1]*mapratio_, iz = ijk[2]*mapratio_;
        // std::cerr << ix << " " << ijk[0] << std::endl;
        if( idim== 0 ){
            return D_xx_.get_cic_kspace({ix,iy,iz});
        }
        else if( idim==1){
            return D_yy_.get_cic_kspace({ix,iy,iz});
        }
        return D_zz_.get_cic_kspace({ix,iy,iz});
    }

    inline ccomplex_t vfac_corr( std::array<size_t,3> ijk ) const
    {
        real_t ix = ijk[0]*mapratio_, iy = ijk[1]*mapratio_, iz = ijk[2]*mapratio_;
        return D_xy_.get_cic_kspace({ix,iy,iz});
    }

};

#if 0
inline void test_plt( void ){

    csoca::ilog << "-------------------------------------------------------------------------------" << std::endl;
    csoca::ilog << "Testing PLT implementation..." << std::endl;

    lattice_gradient lg( 64 );

    return;

    constexpr real_t pi = M_PI, twopi = 2.0*M_PI;

    const std::vector<vec3<real_t>> bcc_normals{
        {0.,pi,pi},{0.,-pi,pi},{0.,pi,-pi},{0.,-pi,-pi},
        {pi,0.,pi},{-pi,0.,pi},{pi,0.,-pi},{-pi,0.,-pi},
        {pi,pi,0.},{-pi,pi,0.},{pi,-pi,0.},{-pi,-pi,0.}
    };

    const std::vector<vec3<real_t>> bcc_reciprocal{
        {twopi,0.,-twopi}, {0.,twopi,-twopi}, {0.,0.,2*twopi}
    };    

    /*const std::vector<vec3<real_t>> fcc_reciprocal{
        {-2.,0.,2.}, {2.,0.,0.}, {1.,1.,-1.}
    };*/

    real_t boxlen = 1.0;
    
    size_t ngrid  = 64;
    size_t npgrid = 1;
    size_t dpg    = ngrid/npgrid;
    size_t nump   = npgrid*npgrid*npgrid;

    real_t pweight = 1.0/real_t(nump);
    real_t eta = 2.0 * boxlen/ngrid;

    const real_t alpha = 1.0/std::sqrt(2)/eta;
    const real_t alpha2 = alpha*alpha;
    const real_t alpha3 = alpha2*alpha;
    const real_t sqrtpi = std::sqrt(M_PI);
    const real_t pi3halfs = std::pow(M_PI,1.5);

    const real_t dV( std::pow( boxlen/ngrid, 3 ) );
    Grid_FFT<real_t> rho({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    auto kronecker = []( int i, int j ) -> real_t { return (i==j)? 1.0 : 0.0; };

    auto greensftide_sr = [&]( int mu, int nu, const vec3<real_t>& vR, const vec3<real_t>& vP ) -> real_t {
        auto d = vR-vP;
        auto r = d.norm();

        if( r< 1e-14 ) return 0.0;

        real_t val = 0.0;
        val -= d[mu]*d[nu]/(r*r) * alpha3/pi3halfs * std::exp(-alpha*alpha*r*r);
        val += 1.0/(4.0*M_PI)*(kronecker(mu,nu)/std::pow(r,3) - 3.0 * (d[mu]*d[nu])/std::pow(r,5)) * 
            (std::erfc(alpha*r) + 2.0*alpha/sqrtpi*std::exp(-alpha*alpha*r*r)*r);
        return pweight * val;
    };

    // sc
    rho.zero();
    rho.relem(0,0,0) = pweight/dV;
    // rho.relem(0,0,0) = pweight/dV/2;
    // rho.relem(ngrid/2,ngrid/2,ngrid/2) = pweight/dV/2;

    rho.FourierTransformForward();
    rho.apply_function_k_dep([&](auto x, auto k) -> ccomplex_t {
        real_t kmod = k.norm();
        std::cerr << x << std::endl;
        return -x * std::exp(-0.5*eta*eta*kmod*kmod) / (kmod*kmod);
    });
    rho.zero_DC_mode();

    auto evaluate_D = [&]( int mu, int nu, const vec3<real_t>& v ) -> real_t{
        real_t sr = 0.0;
        int N = 3;
        for( int i=-N; i<=N; ++i ){
            for( int j=-N; j<=N; ++j ){
                for( int k=-N; k<=N; ++k ){
                    if( std::abs(i)+std::abs(j)+std::abs(k) <= N ){
                        sr += greensftide_sr( mu, nu, v, {real_t(i),real_t(j),real_t(k)} );
                        
                        // sr += greensftide_sr( mu, nu, v, {real_t(i),real_t(j),real_t(k)} )/2;
                        
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
        return sr;
    };

    Grid_FFT<real_t> D_xx({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> D_xy({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> D_xz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> D_yy({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> D_yz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> D_zz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    #pragma omp parallel for
    for( size_t i=0; i<ngrid; i++ ){
        vec3<real_t>  p;
        p.x = real_t(i)/ngrid;
        for( size_t j=0; j<ngrid; j++ ){
            p.y = real_t(j)/ngrid;
            for( size_t k=0; k<ngrid; k++ ){
                p.z = real_t(k)/ngrid;
                D_xx.relem(i,j,k) = evaluate_D(0,0,p);
                D_xy.relem(i,j,k) = evaluate_D(0,1,p);
                D_xz.relem(i,j,k) = evaluate_D(0,2,p);
                D_yy.relem(i,j,k) = evaluate_D(1,1,p);
                D_yz.relem(i,j,k) = evaluate_D(1,2,p);
                D_zz.relem(i,j,k) = evaluate_D(2,2,p);
            }   
        }    
    }

    D_xx.relem(0,0,0) = 0.0;
    D_xy.relem(0,0,0) = 0.0;
    D_xz.relem(0,0,0) = 0.0;
    D_yy.relem(0,0,0) = 0.0;
    D_yz.relem(0,0,0) = 0.0;
    D_zz.relem(0,0,0) = 0.0;

    D_xx.FourierTransformForward();
    D_xy.FourierTransformForward();
    D_xz.FourierTransformForward();
    D_yy.FourierTransformForward();
    D_yz.FourierTransformForward();
    D_zz.FourierTransformForward();

    

    real_t nfac = 1.0/std::pow(real_t(ngrid),1.5);

    real_t kNyquist = M_PI/boxlen * ngrid;

    #pragma omp parallel for
    for( size_t i=0; i<D_xx.size(0); i++ ){
        mat3s<real_t> D;
        vec3<real_t> eval, evec1, evec2, evec3;
        for( size_t j=0; j<D_xx.size(1); j++ ){
            for( size_t k=0; k<D_xx.size(2); k++ ){
                vec3<real_t> kv = D_xx.get_k<real_t>(i,j,k);

                D_xx.kelem(i,j,k) = (D_xx.kelem(i,j,k) - kv[0]*kv[0] * rho.kelem(i,j,k))*nfac + 1.0/3.0;
                D_xy.kelem(i,j,k) = (D_xy.kelem(i,j,k) - kv[0]*kv[1] * rho.kelem(i,j,k))*nfac;
                D_xz.kelem(i,j,k) = (D_xz.kelem(i,j,k) - kv[0]*kv[2] * rho.kelem(i,j,k))*nfac;
                D_yy.kelem(i,j,k) = (D_yy.kelem(i,j,k) - kv[1]*kv[1] * rho.kelem(i,j,k))*nfac + 1.0/3.0;;
                D_yz.kelem(i,j,k) = (D_yz.kelem(i,j,k) - kv[1]*kv[2] * rho.kelem(i,j,k))*nfac;
                D_zz.kelem(i,j,k) = (D_zz.kelem(i,j,k) - kv[2]*kv[2] * rho.kelem(i,j,k))*nfac + 1.0/3.0;;

                D = { std::real(D_xx.kelem(i,j,k)), std::real(D_xy.kelem(i,j,k)), std::real(D_xz.kelem(i,j,k)),
                      std::real(D_yy.kelem(i,j,k)), std::real(D_yz.kelem(i,j,k)), std::real(D_zz.kelem(i,j,k)) };
                
                D.eigen(eval, evec1, evec2, evec3);

                D_xx.kelem(i,j,k) = eval[2];
                D_yy.kelem(i,j,k) = eval[1];
                D_zz.kelem(i,j,k) = eval[0];

                D_xy.kelem(i,j,k) = evec3[0];
                D_xz.kelem(i,j,k) = evec3[1];
                D_yz.kelem(i,j,k) = evec3[2];
            }
        }
    }

#if 1
    std::vector<vec3<real_t>> vectk;
    std::vector<vec3<int>> ico, vecitk;
    vectk.assign(D_xx.size(0)*D_xx.size(1)*D_xx.size(2),vec3<real_t>());
    ico.assign(D_xx.size(0)*D_xx.size(1)*D_xx.size(2),vec3<int>());
    vecitk.assign(D_xx.size(0)*D_xx.size(1)*D_xx.size(2),vec3<int>());

    std::ofstream ofs2("test_brillouin.txt");

    const int numb = 1;
    for( size_t i=0; i<D_xx.size(0); i++ ){
        mat3s<real_t> D;
        vec3<real_t> eval, evec1, evec2, evec3;
        vec3<real_t> a({0.,0.,0.});

        for( size_t j=0; j<D_xx.size(1); j++ ){

            for( size_t k=0; k<D_xx.size(2); k++ ){

                auto idx = D_xx.get_idx(i,j,k);
                vec3<real_t> ar = D_xx.get_k<real_t>(i,j,k) / (twopi*ngrid);
                vec3<real_t> kv = D_xx.get_k<real_t>(i,j,k);
                
                for( int l=0; l<3; l++ ){
                    a[l] = 0.0;
                    for( int m=0; m<3; m++){
                        // project k on reciprocal basis
                        a[l] += ar[m]*bcc_reciprocal[m][l];
                    }
                }

                // translate the k-vectors into the "candidate" FBZ
                vec3<real_t> anum;
                for( int l1=-numb; l1<=numb; ++l1 ){
                    anum[0] = real_t(l1);
                    for( int l2=-numb; l2<=numb; ++l2 ){
                        anum[1] = real_t(l2);
                        for( int l3=-numb; l3<=numb; ++l3 ){
                            anum[2] = real_t(l3);

                            vectk[idx] = a;

                            for( int l=0; l<3; l++ ){
                                for( int m=0; m<3; m++){
                                    // project k on reciprocal basis
                                    vectk[idx][l] += anum[m]*bcc_reciprocal[m][l];
                                }
                            }
                            // check if in first Brillouin zone
                            bool btest=true;
                            for( size_t l=0; l<bcc_normals.size(); ++l ){
                                real_t amod2 = 0.0;
                                real_t scalar = 0.0;
                                for( int m=0; m<3; m++ ){
                                    amod2  += bcc_normals[l][m]*bcc_normals[l][m];
                                    scalar += bcc_normals[l][m]*vectk[idx][m];
                                }
                                real_t amod = std::sqrt(amod2);
                                //if( scalar/amod > amod*1.0001 ){ btest=false; break; }
                                if( scalar > 1.01 * amod2 ){ btest=false; break; }
                            }
                            if( btest ){
                                // int is = (i>ngrid/2)? i-ngrid : i;
                                // int js = (j>ngrid/2)? j-ngrid : j;
                                // int ks = (k>ngrid/2)? k-ngrid : k;
                                
                                vecitk[idx][0] = std::round(vectk[idx][0]*(ngrid)/twopi);
                                vecitk[idx][1] = std::round(vectk[idx][1]*(ngrid)/twopi);
                                vecitk[idx][2] = std::round(vectk[idx][2]*(ngrid)/twopi);

                                ico[idx][0] = std::round((ar[0]+l1) * ngrid);
                                ico[idx][1] = std::round((ar[1]+l2) * ngrid);
                                ico[idx][2] = std::round((ar[2]+l3) * ngrid);

                                assert( std::fabs(real_t(ico[idx][0])/ngrid * bcc_reciprocal[0][0]+real_t(ico[idx][1])/ngrid * bcc_reciprocal[1][0]+real_t(ico[idx][2])/ngrid * bcc_reciprocal[2][0] - vectk[idx][0] ) < 1e-12 );
                                assert( std::fabs(real_t(ico[idx][0])/ngrid * bcc_reciprocal[0][1]+real_t(ico[idx][1])/ngrid * bcc_reciprocal[1][1]+real_t(ico[idx][2])/ngrid * bcc_reciprocal[2][1] - vectk[idx][1] ) < 1e-12 );
                                assert( std::fabs(real_t(ico[idx][0])/ngrid * bcc_reciprocal[0][2]+real_t(ico[idx][1])/ngrid * bcc_reciprocal[1][2]+real_t(ico[idx][2])/ngrid * bcc_reciprocal[2][2] - vectk[idx][2] ) < 1e-12 );
                                

                                
                                // if( ico[idx][2] < 0 ){
                                //     ico[idx][0] = -ico[idx][0];
                                //     ico[idx][1] = -ico[idx][1];
                                //     ico[idx][2] = -ico[idx][2];
                                // }

                                // ico[idx][0] = (ico[idx][0]+ngrid)%ngrid;
                                // ico[idx][1] = (ico[idx][1]+ngrid)%ngrid;

                                // if( vectk[idx][2] < 0 ){
                                //     vectk[idx][0] = - vectk[idx][0];
                                //     vectk[idx][1] = - vectk[idx][1];
                                //     vectk[idx][2] = - vectk[idx][2];
                                // }

                                // if( vecitk[idx][2] < 0 ){
                                //     vecitk[idx][0] = -vecitk[idx][0];
                                //     vecitk[idx][1] = -vecitk[idx][1];
                                //     vecitk[idx][2] = -vecitk[idx][2];
                                // }
                                //vecitk[idx][0] = (vecitk[idx][0]+ngrid)%ngrid;
                                //vecitk[idx][1] = (vecitk[idx][1]+ngrid)%ngrid;
                                //vecitk[idx][2] = (vecitk[idx][2]+ngrid)%ngrid;
                                
                                

                                //vecitk[idx][0] = (vecitk[idx][0]<0)? vecitk[idx][0]+ngrid : vecitk[idx][0];;
                                //vecitk[idx][1] = (vecitk[idx][1]<0)? vecitk[idx][1]+ngrid : vecitk[idx][1];
                                


                                //ofs2 << kv.x << ", " << kv.y << ", " << kv.z << ", " << vectk[idx].x*(ngrid)/twopi << ", " << vectk[idx].y*(ngrid)/twopi << ", " << vectk[idx].z*(ngrid)/twopi << ", " << ico[idx][0] << ", " << ico[idx][1] << ", " << ico[idx][2] << std::endl;
                                ofs2 << kv.x/twopi << ", " << kv.y/twopi << ", " << kv.z/twopi << ", " << vecitk[idx].x << ", " << vecitk[idx].y << ", " << vecitk[idx].z << ", " << ico[idx][0] << ", " << ico[idx][1] << ", " << ico[idx][2] << std::endl;
                                ofs2 << kv.x/twopi << ", " << kv.y/twopi << ", " << kv.z/twopi << ", " << -vecitk[idx].x << ", " << -vecitk[idx].y << ", " << -vecitk[idx].z << ", " << ico[idx][0] << ", " << ico[idx][1] << ", " << ico[idx][2] << std::endl;
                                
                                // std::cerr << real_t(ico[idx][0])/ngrid * bcc_reciprocal[0][0]+real_t(ico[idx][1])/ngrid * bcc_reciprocal[1][0]+real_t(ico[idx][2])/ngrid * bcc_reciprocal[2][0] << " " <<  vectk[idx][0] << std::endl;
                                
                                // std::cerr << real_t(ico[idx][0])/ngrid * bcc_reciprocal[0][0]+real_t(ico[idx][1])/ngrid * bcc_reciprocal[1][0]+real_t(ico[idx][2])/ngrid * bcc_reciprocal[2][0] << " " <<  vectk[idx][0] << std::endl;
                                //std::cerr << real_t(ico[idx][0])/ngrid * bcc_reciprocal[0][1]+real_t(ico[idx][1])/ngrid * bcc_reciprocal[1][1]+real_t(ico[idx][2])/ngrid * bcc_reciprocal[2][1] << " " <<  vectk[idx][1] << std::endl;
                                // assert( std::fabs(real_t(ico[idx][0])/ngrid * bcc_reciprocal[0][1]+real_t(ico[idx][1])/ngrid * bcc_reciprocal[1][1]+real_t(ico[idx][2])/ngrid * bcc_reciprocal[2][1] - vectk[idx][1] ) < 1e-12 );
                                goto endloop;
                            }
                        }
                    }
                }
                endloop: ;

                //D_xx.kelem(i,j,k) = D_xx.kelem(ico[idx][0],ico[idx][1],ico[idx][2]);
                // D_xx.kelem(ico[idx][0],ico[idx][1],ico[idx][2]) = D_xx.kelem(i,j,k);
                //D_xx.kelem(i,j,k) = D_xx.kelem(i+vecitk[idx][0],j+vecitk[idx][1],k+vecitk[idx][2]);
            }
        }
            
    }

#endif

    std::ofstream ofs("test_ewald.txt");
    for( size_t i=0; i<D_xx.size(0); i++ ){
        for( size_t j=0; j<D_xx.size(1); j++ ){
            for( size_t k=0; k<D_xx.size(2); k++ ){
                vec3<real_t> kv = D_xx.get_k<real_t>(i,j,k);
                ofs << std::setw(16) << kv.norm() / kNyquist
                    << std::setw(16) << std::real(D_xx.kelem(i,j,k))
                    << std::setw(16) << std::real(D_yy.kelem(i,j,k))
                    << std::setw(16) << std::real(D_zz.kelem(i,j,k))
                    << std::setw(16) << kv[0]
                    << std::setw(16) << kv[1]
                    << std::setw(16) << kv[2]
                    << std::endl;
            }
        }
    }


    std::string filename("plt_test.hdf5");
    unlink(filename.c_str());
#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
//     rho.Write_to_HDF5(filename, "rho");
    D_xx.Write_to_HDF5(filename, "omega1");
    D_yy.Write_to_HDF5(filename, "omega2");
    D_zz.Write_to_HDF5(filename, "omega3");
    D_xy.Write_to_HDF5(filename, "e1_x");
    D_xz.Write_to_HDF5(filename, "e1_y");
    D_yz.Write_to_HDF5(filename, "e1_z");

}
#endif

}