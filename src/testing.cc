#include <testing.hh>

void output_potentials_and_densities(
    size_t ngrid, real_t boxlen,
    const Grid_FFT<real_t> &phi,
    const Grid_FFT<real_t> &phi2,
    const Grid_FFT<real_t> &phi3a,
    const Grid_FFT<real_t> &phi3b,
    const std::array<Grid_FFT<real_t> *, 3> &A3)
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