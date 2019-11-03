#pragma once

#include <general.hh>
#include <unistd.h> // for unlink

#include <iostream>
#include <fstream>

#include <random>

#include <mat3.hh>

namespace particle{
//! implement Marcos et al. PLT calculation

inline void test_plt( void ){

    csoca::ilog << "-------------------------------------------------------------------------------" << std::endl;
    csoca::ilog << "Testing PLT implementation..." << std::endl;

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
    std::vector< vec3<real_t> > gpos ;

    auto kronecker = []( int i, int j ) -> real_t { return (i==j)? 1.0 : 0.0; };

    auto greensftide_sr = [&]( int mu, int nu, const vec3<real_t>& vR, const vec3<real_t>& vP ) -> real_t {
        auto d = vR-vP;
        d.x = (d.x>0.5)? d.x-1.0 : (d.x<-0.5)? d.x+1.0 : d.x;
        d.y = (d.y>0.5)? d.y-1.0 : (d.y<-0.5)? d.y+1.0 : d.y;
        d.z = (d.z>0.5)? d.z-1.0 : (d.z<-0.5)? d.z+1.0 : d.z;
        auto r = d.norm();

        if( r< 1e-14 ) return 0.0;

        real_t val = 0.0;

        val -= d[mu]*d[nu]/(r*r) * alpha3/pi3halfs * std::exp(-alpha*alpha*r*r);
        val += 1.0/(4.0*M_PI)*(kronecker(mu,nu)/std::pow(r,3) - 3.0 * (d[mu]*d[nu])/std::pow(r,5)) * 
            (std::erfc(alpha*r) + 2.0*alpha/sqrtpi*std::exp(-alpha*alpha*r*r)*r);

        return pweight * val;
    };

    gpos.reserve(nump);

    // sc
    for( size_t i=0; i<npgrid; ++i ){
        for( size_t j=0; j<npgrid; ++j ){
            for( size_t k=0; k<npgrid; ++k ){
                rho.relem(i*dpg,j*dpg,k*dpg) = pweight/dV;
                gpos.push_back({real_t(i)/npgrid,real_t(j)/npgrid,real_t(k)/npgrid});
            }
        }    
    }

    rho.FourierTransformForward();
    rho.apply_function_k_dep([&](auto x, auto k) -> ccomplex_t {
        real_t kmod = k.norm();
        return -x * std::exp(-0.5*eta*eta*kmod*kmod) / (kmod*kmod);
    });
    rho.zero_DC_mode();

    auto evaluate_D = [&]( int mu, int nu, const vec3<real_t>& v ) -> real_t{
        real_t sr = 0.0;
        for( auto& p : gpos ){
            sr += greensftide_sr( mu, nu, v, p);
        }
        if( v.norm()<1e-14 ) return 0.0;

        return sr;
    };


    // std::random_device rd;  //Will be used to obtain a seed for the random number engine
    // std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    // std::uniform_real_distribution<> dis(-0.25,0.25);

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

                //D = {evaluate_D(0,0,p),evaluate_D(0,1,p),evaluate_D(0,2,p),evaluate_D(1,0,p),evaluate_D(1,1,p),evaluate_D(2,2,p)};
                //D.eigen(eval, evec1, evec2, evec3);
                //rho.relem(i,j,k) = eval[2];
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

    std::ofstream ofs("test_ewald.txt");

    real_t nfac = 1.0/std::pow(real_t(ngrid),1.5);

    real_t kNyquist = M_PI/boxlen * ngrid;

    //#pragma omp parallel for
    for( size_t i=0; i<D_xx.size(0); i++ ){
        mat3s<real_t> D;
        vec3<real_t> eval, evec1, evec2, evec3;
        for( size_t j=0; j<D_xx.size(1); j++ ){
            for( size_t k=0; k<D_xx.size(2); k++ ){
                vec3<real_t> kv = D_xx.get_k<real_t>(i,j,k);

                D = { std::real(D_xx.kelem(i,j,k) - kv[0]*kv[0] * rho.kelem(i,j,k) ),
                      std::real(D_xy.kelem(i,j,k) - kv[0]*kv[1] * rho.kelem(i,j,k) ),
                      std::real(D_xz.kelem(i,j,k) - kv[0]*kv[2] * rho.kelem(i,j,k) ),
                      std::real(D_yy.kelem(i,j,k) - kv[1]*kv[1] * rho.kelem(i,j,k) ),
                      std::real(D_yz.kelem(i,j,k) - kv[1]*kv[2] * rho.kelem(i,j,k) ),
                      std::real(D_zz.kelem(i,j,k) - kv[2]*kv[2] * rho.kelem(i,j,k) ) };
                D.eigen(eval, evec1, evec2, evec3);
                

                ofs << std::setw(16) << kv.norm() / kNyquist
                    << std::setw(16) << eval[0] *nfac + 1.0/3.0
                    << std::setw(16) << eval[1] *nfac + 1.0/3.0
                    << std::setw(16) << eval[2] *nfac + 1.0/3.0
                    << std::setw(16) << kv[0]
                    << std::setw(16) << kv[1]
                    << std::setw(16) << kv[2]
                    << std::endl;
            }
        }
    }

//     std::string filename("plt_test.hdf5");
//     unlink(filename.c_str());
// #if defined(USE_MPI)
//     MPI_Barrier(MPI_COMM_WORLD);
// #endif
//     rho.Write_to_HDF5(filename, "rho");

}


}