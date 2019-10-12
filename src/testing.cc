#include <testing.hh>
#include <unistd.h> // for unlink

#include <operators.hh>
#include <convolution.hh>

namespace testing
{

void output_potentials_and_densities(
    ConfigFile &the_config,
    size_t ngrid, real_t boxlen,
    Grid_FFT<real_t> &phi,
    Grid_FFT<real_t> &phi2,
    Grid_FFT<real_t> &phi3a,
    Grid_FFT<real_t> &phi3b,
    std::array<Grid_FFT<real_t> *, 3> &A3)
{
    const std::string fname_hdf5 = the_config.GetValueSafe<std::string>("output", "fname_hdf5", "output.hdf5");
    const std::string fname_analysis = the_config.GetValueSafe<std::string>("output", "fbase_analysis", "output");

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
    ConfigFile &the_config,
    size_t ngrid, real_t boxlen, real_t vfac, real_t dplus,
    Grid_FFT<real_t> &phi,
    Grid_FFT<real_t> &phi2,
    Grid_FFT<real_t> &phi3a,
    Grid_FFT<real_t> &phi3b,
    std::array<Grid_FFT<real_t> *, 3> &A3,
    bool bwrite_out_fields)
{
    const std::string fname_hdf5 = the_config.GetValueSafe<std::string>("output", "fname_hdf5", "output.hdf5");
    const std::string fname_analysis = the_config.GetValueSafe<std::string>("output", "fbase_analysis", "output");

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
                    auto kk = phi.get_k<real_t>(i, j, k);
                    size_t idx = phi.get_idx(i, j, k);
                    auto phitot = phi.kelem(idx) + phi2.kelem(idx) + phi3a.kelem(idx) + phi3b.kelem(idx);
                    auto phitot_v = vfac1 * phi.kelem(idx) + vfac2 * phi2.kelem(idx) + vfac3 * (phi3a.kelem(idx) + phi3b.kelem(idx));

                    // divide by Lbox, because displacement is in box units for output plugin
                    grid_x[idim]->kelem(idx) = ccomplex_t(0.0, -1.0) * (kk[idim] * phitot + kk[idimp] * A3[idimpp]->kelem(idx) - kk[idimpp] * A3[idimp]->kelem(idx));
                    grid_v[idim]->kelem(idx) = ccomplex_t(0.0, -1.0) * (kk[idim] * phitot_v + vfac3 * (kk[idimp] * A3[idimpp]->kelem(idx) - kk[idimpp] * A3[idimp]->kelem(idx))) / boxlen;
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
                    auto kk = invariant.get_k<real_t>(i, j, k);
                    size_t idx = invariant.get_idx(i, j, k);
                    invariant.kelem(idx) = ccomplex_t(0.0, -kk[idimp]) * grid_v[idim]->kelem(idx) - ccomplex_t(0.0, -kk[idim]) * grid_v[idimp]->kelem(idx);
                    // invariant.kelem(idx) = ccomplex_t(0.0,kk[idim]) * grid_v[idimp]->kelem(idx)
                    //                      - ccomplex_t(0.0,kk[idimp])  * grid_v[idim]->kelem(idx);
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


    std::cerr << "std. deviation of invariant : ( D+ | I_xy | I_yz | I_zx ) \n"
                    << std::setw(16) << dplus << " "
                    << std::setw(16) << Icomp[0] << " "
                    << std::setw(16) << Icomp[1] << " "
                    << std::setw(16) << Icomp[2] << std::endl;

}

} // namespace testing