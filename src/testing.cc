#include <testing.hh>
#include <unistd.h> // for unlink
#include <memory>

#include <operators.hh>
#include <convolution.hh>

namespace testing
{

void output_potentials_and_densities(
    config_file &the_config,
    size_t ngrid, real_t boxlen,
    Grid_FFT<real_t> &phi,
    Grid_FFT<real_t> &phi2,
    Grid_FFT<real_t> &phi3a,
    Grid_FFT<real_t> &phi3b,
    std::array<Grid_FFT<real_t> *, 3> &A3)
{
    const std::string fname_hdf5 = the_config.get_value_safe<std::string>("output", "fname_hdf5", "output.hdf5");
    const std::string fname_analysis = the_config.get_value_safe<std::string>("output", "fbase_analysis", "output");

    Grid_FFT<real_t> delta({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> delta2({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> delta3a({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> delta3b({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> delta3({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    delta.FourierTransformForward(false);
    delta2.FourierTransformForward(false);
    delta3a.FourierTransformForward(false);
    delta3b.FourierTransformForward(false);
    delta3.FourierTransformForward(false);

#pragma omp parallel for
    for (size_t i = 0; i < phi.size(0); ++i)
    {
        for (size_t j = 0; j < phi.size(1); ++j)
        {
            for (size_t k = 0; k < phi.size(2); ++k)
            {
                auto kk = phi.get_k<real_t>(i, j, k);
                size_t idx = phi.get_idx(i, j, k);
                auto laplace = -kk.norm_squared();

                // compute densities associated to respective potentials as well
                delta.kelem(idx) = laplace * phi.kelem(idx);
                delta2.kelem(idx) = laplace * phi2.kelem(idx);
                delta3a.kelem(idx) = laplace * phi3a.kelem(idx);
                delta3b.kelem(idx) = laplace * phi3b.kelem(idx);
                delta3.kelem(idx) = delta3a.kelem(idx) + delta3b.kelem(idx);
            }
        }
    }

    delta.Write_PowerSpectrum(fname_analysis + "_" + "power_delta1.txt");
    delta2.Write_PowerSpectrum(fname_analysis + "_" + "power_delta2.txt");
    delta3a.Write_PowerSpectrum(fname_analysis + "_" + "power_delta3a.txt");
    delta3b.Write_PowerSpectrum(fname_analysis + "_" + "power_delta3b.txt");
    delta3.Write_PowerSpectrum(fname_analysis + "_" + "power_delta3.txt");

    phi.FourierTransformBackward();
    phi2.FourierTransformBackward();
    phi3a.FourierTransformBackward();
    phi3b.FourierTransformBackward();

    delta.FourierTransformBackward();
    delta2.FourierTransformBackward();
    delta3a.FourierTransformBackward();
    delta3b.FourierTransformBackward();
    delta3.FourierTransformBackward();

    A3[0]->FourierTransformBackward();
    A3[1]->FourierTransformBackward();
    A3[2]->FourierTransformBackward();

#if defined(USE_MPI)
    if (CONFIG::MPI_task_rank == 0)
        unlink(fname_hdf5.c_str());
    MPI_Barrier(MPI_COMM_WORLD);
#else
    unlink(fname_hdf5.c_str());
#endif

    phi.Write_to_HDF5(fname_hdf5, "phi");
    phi2.Write_to_HDF5(fname_hdf5, "phi2");
    phi3a.Write_to_HDF5(fname_hdf5, "phi3a");
    phi3b.Write_to_HDF5(fname_hdf5, "phi3b");

    delta.Write_to_HDF5(fname_hdf5, "delta");
    delta2.Write_to_HDF5(fname_hdf5, "delta2");
    delta3a.Write_to_HDF5(fname_hdf5, "delta3a");
    delta3b.Write_to_HDF5(fname_hdf5, "delta3b");
    delta3.Write_to_HDF5(fname_hdf5, "delta3");

    A3[0]->Write_to_HDF5(fname_hdf5, "A3x");
    A3[1]->Write_to_HDF5(fname_hdf5, "A3y");
    A3[2]->Write_to_HDF5(fname_hdf5, "A3z");
}

void output_velocity_displacement_symmetries(
    config_file &the_config,
    size_t ngrid, real_t boxlen, real_t vfac, real_t dplus,
    Grid_FFT<real_t> &phi,
    Grid_FFT<real_t> &phi2,
    Grid_FFT<real_t> &phi3a,
    Grid_FFT<real_t> &phi3b,
    std::array<Grid_FFT<real_t> *, 3> &A3,
    bool bwrite_out_fields)
{
    const std::string fname_hdf5 = the_config.get_value_safe<std::string>("output", "fname_hdf5", "output.hdf5");
    const std::string fname_analysis = the_config.get_value_safe<std::string>("output", "fbase_analysis", "output");

    real_t vfac1 = vfac;
    real_t vfac2 = 2 * vfac;
    real_t vfac3 = 3 * vfac;

    Grid_FFT<real_t> px({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> py({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> pz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> vx({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> vy({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> vz({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    //... array [.] access to components of A3:
    std::array<Grid_FFT<real_t> *, 3> grid_v({&vx, &vy, &vz});
    std::array<Grid_FFT<real_t> *, 3> grid_x({&px, &py, &pz});

#if defined(USE_MPI)
    if (CONFIG::MPI_task_rank == 0)
        unlink(fname_hdf5.c_str());
    MPI_Barrier(MPI_COMM_WORLD);
#else
    unlink(fname_hdf5.c_str());
#endif

    for (int idim = 0; idim < 3; ++idim)
    {
        // cyclic rotations of indices
        const int idimp = (idim + 1) % 3, idimpp = (idim + 2) % 3;
        grid_x[idim]->zero();
        grid_v[idim]->zero();

        grid_x[idim]->FourierTransformForward(false);
        grid_v[idim]->FourierTransformForward(false);

        // combine the various LPT potentials into one and take gradient
        #pragma omp parallel for
        for (size_t i = 0; i < phi.size(0); ++i)
        {
            for (size_t j = 0; j < phi.size(1); ++j)
            {
                for (size_t k = 0; k < phi.size(2); ++k)
                {
                    size_t idx = phi.get_idx(i, j, k);
                    auto phitot = phi.kelem(idx) + phi2.kelem(idx) + phi3a.kelem(idx) + phi3b.kelem(idx);
                    auto phitot_v = vfac1 * phi.kelem(idx) + vfac2 * phi2.kelem(idx) + vfac3 * (phi3a.kelem(idx) + phi3b.kelem(idx));

                    // divide by Lbox, because displacement is in box units for output plugin
                    grid_x[idim]->kelem(idx) = ( phi.gradient(idim,{i,j,k}) * phitot 
                                                + phi.gradient(idimp,{i,j,k}) * A3[idimpp]->kelem(idx) 
                                                - phi.gradient(idimpp,{i,j,k}) * A3[idimp]->kelem(idx) );
                    grid_v[idim]->kelem(idx) = ( phi.gradient(idim,{i,j,k}) * phitot_v 
                            + vfac3 * (phi.gradient(idimp,{i,j,k}) * A3[idimpp]->kelem(idx) - phi.gradient(idimpp,{i,j,k}) * A3[idimp]->kelem(idx)) ) / boxlen;
                }
            }
        }
        if (bwrite_out_fields)
        {
            grid_x[idim]->FourierTransformBackward();
            grid_v[idim]->FourierTransformBackward();
        }
    }

    if (bwrite_out_fields)
    {
        grid_x[0]->Write_to_HDF5(fname_hdf5, "px");
        grid_x[1]->Write_to_HDF5(fname_hdf5, "py");
        grid_x[2]->Write_to_HDF5(fname_hdf5, "pz");
        grid_v[0]->Write_to_HDF5(fname_hdf5, "vx");
        grid_v[1]->Write_to_HDF5(fname_hdf5, "vy");
        grid_v[2]->Write_to_HDF5(fname_hdf5, "vz");

        for (int idim = 0; idim < 3; ++idim)
        {
            grid_x[idim]->FourierTransformForward();
            grid_v[idim]->FourierTransformForward();
        }
    }

    OrszagConvolver<real_t> Conv({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> invariant({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    std::array<double,3> Icomp;

    for (int idim = 0; idim < 3; ++idim)
    {
        int idimp = (idim + 1) % 3;

        // dx_k/dq_i * dv_k/dq_j = (delta_ki + dPsi_k/dq_i) dv_k / dq_j = dv_i / dq_j + dPsi_k/dq_i *dv_k/dq_j
        // dx_k/dq_j * dv_k/dq_i = dv_j / dq_i + dPsi_k/dq_j *dv_k/dq_i
        // invariant = dv_i/dq_j-dv_j/dq_i + dPsi_k/dq_i *dv_k/dq_j - dPsi_k/dq_j *dv_k/dq_i
        invariant.FourierTransformForward(false);

        for (size_t i = 0; i < invariant.size(0); ++i)
        {
            for (size_t j = 0; j < invariant.size(1); ++j)
            {
                for (size_t k = 0; k < invariant.size(2); ++k)
                {
                    size_t idx = invariant.get_idx(i, j, k);
                   invariant.kelem(idx) = invariant.gradient(idimp,{i,j,k}) * grid_v[idim]->kelem(idx) - invariant.gradient(idim,{i,j,k}) * grid_v[idimp]->kelem(idx);

                }
            }
        }

        Conv.convolve_Gradients(px, {idim}, vx, {idimp}, op::add_to(invariant));
        Conv.convolve_Gradients(py, {idim}, vy, {idimp}, op::add_to(invariant));
        Conv.convolve_Gradients(pz, {idim}, vz, {idimp}, op::add_to(invariant));
        Conv.convolve_Gradients(px, {idimp}, vx, {idim}, op::subtract_from(invariant));
        Conv.convolve_Gradients(py, {idimp}, vy, {idim}, op::subtract_from(invariant));
        Conv.convolve_Gradients(pz, {idimp}, vz, {idim}, op::subtract_from(invariant));
        invariant.FourierTransformBackward();

        if (bwrite_out_fields)
        {
            char fdescr[32];
            sprintf(fdescr, "inv%d%d", idim, idimp);
            invariant.Write_to_HDF5(fname_hdf5, fdescr);
        }
        
        Icomp[idim] = invariant.std();
    }


    music::ilog << "std. deviation of invariant : ( D+ | I_xy | I_yz | I_zx ) \n"
                    << std::setw(16) << dplus << " "
                    << std::setw(16) << Icomp[0] << " "
                    << std::setw(16) << Icomp[1] << " "
                    << std::setw(16) << Icomp[2] << std::endl;

}

void output_convergence(
    config_file &the_config,
    cosmology::calculator* the_cosmo_calc,
    std::size_t ngrid, real_t boxlen, real_t vfac, real_t dplus,
    Grid_FFT<real_t> &phi,
    Grid_FFT<real_t> &phi2,
    Grid_FFT<real_t> &phi3a,
    Grid_FFT<real_t> &phi3b,
    std::array<Grid_FFT<real_t> *, 3> &A3)
{
    // scale all potentials to remove dplus0
    phi /= dplus;
    phi2 /= dplus * dplus;
    phi3a /= dplus * dplus * dplus;
    phi3b /= dplus * dplus * dplus;
    (*A3[0]) /= dplus * dplus * dplus;
    (*A3[1]) /= dplus * dplus * dplus;
    (*A3[2]) /= dplus * dplus * dplus;

    ////////////////////// theoretical convergence radius //////////////////////

    // compute phi_code
    Grid_FFT<real_t> phi_code({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    phi_code.FourierTransformForward(false);
    #pragma omp parallel for //collapse(3)
    for (std::size_t i = 0; i < phi_code.size(0); ++i) {
        for (std::size_t j = 0; j < phi_code.size(1); ++j) {
            for (std::size_t k = 0; k < phi_code.size(2); ++k) {
                std::size_t idx = phi_code.get_idx(i, j, k);
                phi_code.kelem(idx) = -phi.kelem(idx);
            }
        }
    }

    // initialize norm to 0
    Grid_FFT<real_t> nabla_vini_norm({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    #pragma omp parallel for //collapse(3)
    for (std::size_t i = 0; i < nabla_vini_norm.size(0); ++i) {
        for (std::size_t j = 0; j < nabla_vini_norm.size(1); ++j) {
            for (std::size_t k = 0; k < nabla_vini_norm.size(2); ++k) {
                std::size_t idx = nabla_vini_norm.get_idx(i, j, k);
                nabla_vini_norm.relem(idx) = 0.0;
            }
        }
    }

    Grid_FFT<real_t> nabla_vini_mn({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    for(std::size_t m = 0; m < 3; m++) {
        for(std::size_t n = m; n < 3; n++) {
            nabla_vini_mn.FourierTransformForward(false);
            #pragma omp parallel for //collapse(3)
            for (std::size_t i = 0; i < phi_code.size(0); ++i) {
                for (std::size_t j = 0; j < phi_code.size(1); ++j) {
                    for (std::size_t k = 0; k < phi_code.size(2); ++k) {
                        std::size_t idx = phi_code.get_idx(i, j, k);
                        auto kk = phi_code.get_k<real_t>(i, j, k);
                        nabla_vini_mn.kelem(idx) = phi_code.kelem(idx) * (kk[m] * kk[n]);
                    }
                }
            }
            nabla_vini_mn.FourierTransformBackward();
            nabla_vini_mn *= (3.2144004915 / the_cosmo_calc->get_growth_factor(1.0));
            // sum of squares
            #pragma omp parallel for //collapse(3)
            for (std::size_t i = 0; i < nabla_vini_norm.size(0); ++i) {
                for (std::size_t j = 0; j < nabla_vini_norm.size(1); ++j) {
                    for (std::size_t k = 0; k < nabla_vini_norm.size(2); ++k) {
                        std::size_t idx = nabla_vini_norm.get_idx(i, j, k);
                        if(m != n) {
                            nabla_vini_norm.relem(idx) += (2.0 * nabla_vini_mn.relem(idx) * nabla_vini_mn.relem(idx));
                        } else {
                            nabla_vini_norm.relem(idx) += (nabla_vini_mn.relem(idx) * nabla_vini_mn.relem(idx));
                        }
                    }
                }
            }
        }
    }
    // square root
    #pragma omp parallel for //collapse(3)
    for (std::size_t i = 0; i < nabla_vini_norm.size(0); ++i) {
        for (std::size_t j = 0; j < nabla_vini_norm.size(1); ++j) {
            for (std::size_t k = 0; k < nabla_vini_norm.size(2); ++k) {
                std::size_t idx = nabla_vini_norm.get_idx(i, j, k);
                nabla_vini_norm.relem(idx) = std::sqrt(nabla_vini_norm.relem(idx));
            }
        }
    }

    // get t_eds
    Grid_FFT<real_t> t_eds({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    #pragma omp parallel for //collapse(3)
    for (std::size_t i = 0; i < t_eds.size(0); ++i) {
        for (std::size_t j = 0; j < t_eds.size(1); ++j) {
            for (std::size_t k = 0; k < t_eds.size(2); ++k) {
                std::size_t idx = t_eds.get_idx(i, j, k);
                t_eds.relem(idx) = 0.0204 / nabla_vini_norm.relem(idx);
            }
        }
    }

    ////////////////////////// 3lpt convergence test ///////////////////////////

    // initialize grids to 0
    Grid_FFT<real_t> psi_1({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> psi_2({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> psi_3({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    #pragma omp parallel for //collapse(3)
    for (std::size_t i = 0; i < psi_1.size(0); ++i) {
        for (std::size_t j = 0; j < psi_1.size(1); ++j) {
            for (std::size_t k = 0; k < psi_1.size(2); ++k) {
                std::size_t idx = psi_1.get_idx(i, j, k);
                psi_1.relem(idx) = 0.0;
                psi_2.relem(idx) = 0.0;
                psi_3.relem(idx) = 0.0;
            }
        }
    }


    // temporaries
    Grid_FFT<real_t> psi_1_tmp({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> psi_2_tmp({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> psi_3_tmp({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    // compute psi 1, 2 and 3
    for (int idim = 0; idim < 3; ++idim) {
        // cyclic rotations of indices
        int idimp = (idim + 1) % 3, idimpp = (idim + 2) % 3;

        psi_1_tmp.FourierTransformForward(false);
        psi_2_tmp.FourierTransformForward(false);
        psi_3_tmp.FourierTransformForward(false);

        #pragma omp parallel for //collapse(3)
        for (std::size_t i = 0; i < phi.size(0); ++i) {
            for (std::size_t j = 0; j < phi.size(1); ++j) {
                for (std::size_t k = 0; k < phi.size(2); ++k) {
                    auto kk = phi.get_k<real_t>(i, j, k);
                    std::size_t idx = phi.get_idx(i, j, k);
                    psi_1_tmp.kelem(idx) = ccomplex_t(0.0, 1.0) * (kk[idim] * phi.kelem(idx));
                    psi_2_tmp.kelem(idx) = ccomplex_t(0.0, 1.0) * (kk[idim] * phi2.kelem(idx));
                    psi_3_tmp.kelem(idx) = ccomplex_t(0.0, 1.0) * (
                        kk[idim] * (phi3a.kelem(idx) + phi3b.kelem(idx)) +
                        kk[idimp] * A3[idimpp]->kelem(idx) -
                        kk[idimpp] * A3[idimp]->kelem(idx)
                    );
                }
            }
        }
        psi_1_tmp.FourierTransformBackward();
        psi_2_tmp.FourierTransformBackward();
        psi_3_tmp.FourierTransformBackward();

        // sum of squares
        #pragma omp parallel for //collapse(3)
        for (std::size_t i = 0; i < psi_1.size(0); ++i) {
            for (std::size_t j = 0; j < psi_1.size(1); ++j) {
                for (std::size_t k = 0; k < psi_1.size(2); ++k) {
                    std::size_t idx = psi_1.get_idx(i, j, k);
                    psi_1.relem(idx) += psi_1_tmp.relem(idx) * psi_1_tmp.relem(idx);
                    psi_2.relem(idx) += psi_2_tmp.relem(idx) * psi_2_tmp.relem(idx);
                    psi_3.relem(idx) += psi_3_tmp.relem(idx) * psi_3_tmp.relem(idx);
                }
            }
        }
    } // loop on dimensions

    // apply square root for the L2 norm
#pragma omp parallel for //collapse(3)
    for (std::size_t i = 0; i < psi_1.size(0); ++i) {
        for (std::size_t j = 0; j < psi_1.size(1); ++j) {
            for (std::size_t k = 0; k < psi_1.size(2); ++k) {
                std::size_t idx = psi_1.get_idx(i, j, k);
                psi_1.relem(idx) = std::sqrt(psi_1.relem(idx));
                psi_2.relem(idx) = std::sqrt(psi_2.relem(idx));
                psi_3.relem(idx) = std::sqrt(psi_3.relem(idx));
            }
        }
    }

    // convergence radius
    Grid_FFT<real_t> inv_convergence_radius({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    #pragma omp parallel for //collapse(3)
    for (std::size_t i = 0; i < psi_1.size(0); ++i) {
        for (std::size_t j = 0; j < psi_1.size(1); ++j) {
            for (std::size_t k = 0; k < psi_1.size(2); ++k) {
                std::size_t idx = psi_1.get_idx(i, j, k);
                inv_convergence_radius.relem(idx) =
                    3.0 * (std::abs(psi_3.relem(idx)) / std::abs(psi_2.relem(idx))) -
                    2.0 * (std::abs(psi_2.relem(idx)) / std::abs(psi_1.relem(idx)));
            }
        }
    }

    ////////////////////////////// write results ///////////////////////////////
    std::string convergence_test_filename("convergence_test.hdf5");
    unlink(convergence_test_filename.c_str());
#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    t_eds.Write_to_HDF5(convergence_test_filename, "t_eds");
    inv_convergence_radius.Write_to_HDF5(convergence_test_filename, "inv_convergence_radius");
    // psi_1.Write_to_HDF5(convergence_test_filename, "psi_1_norm");
    // psi_2.Write_to_HDF5(convergence_test_filename, "psi_2_norm");
    // psi_3.Write_to_HDF5(convergence_test_filename, "psi_3_norm");
}

} // namespace testing
