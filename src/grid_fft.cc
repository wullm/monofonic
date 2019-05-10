#include <general.hh>
#include <grid_fft.hh>
#include <thread>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

template <typename data_t>
void Grid_FFT<data_t>::FillRandomReal( unsigned long int seed )
{
    gsl_rng *RNG = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(RNG, seed);

    for (size_t i = 0; i < sizes_[0]; ++i)
    {
        for (size_t j = 0; j < sizes_[1]; ++j)
        {
            for (size_t k = 0; k < sizes_[2]; ++k)
            {
                this->relem(i,j,k) = gsl_ran_ugaussian_ratio_method(RNG);
            }
        }
    }

    gsl_rng_free(RNG);
}

template <typename data_t>
void Grid_FFT<data_t>::Setup(void)
{
    if (CONFIG::FFTW_threads_ok)
        fftw_plan_with_nthreads(std::thread::hardware_concurrency());

#if !defined(USE_MPI) ////////////////////////////////////////////////////////////////////////////////////////////

    ntot_ = (n_[2] + 2) * n_[1] * n_[0];

    
    csoca::ilog.Print("[FFT] Setting up a shared memory field %lux%lux%lu\n", n_[0], n_[1], n_[2]);
    if (typeid(data_t) == typeid(real_t))
    {
        data_ = reinterpret_cast<data_t *>(fftw_malloc(ntot_ * sizeof(real_t)));
        cdata_ = reinterpret_cast<ccomplex_t *>(data_);

        plan_ = FFTW_API(plan_dft_r2c_3d)(n_[0], n_[1], n_[2], (real_t *)data_, (complex_t *)data_, FFTW_RUNMODE);
        iplan_ = FFTW_API(plan_dft_c2r_3d)(n_[0], n_[1], n_[2], (complex_t *)data_, (real_t *)data_, FFTW_RUNMODE);
    }
    else if (typeid(data_t) == typeid(ccomplex_t))
    {
        data_ = reinterpret_cast<data_t *>(fftw_malloc(ntot_ * sizeof(ccomplex_t)));
        cdata_ = reinterpret_cast<ccomplex_t *>(data_);

        plan_ = FFTW_API(plan_dft_3d)(n_[0], n_[1], n_[2], (complex_t *)data_, (complex_t *)data_, FFTW_FORWARD, FFTW_RUNMODE);
        iplan_ = FFTW_API(plan_dft_3d)(n_[0], n_[1], n_[2], (complex_t *)data_, (complex_t *)data_, FFTW_BACKWARD, FFTW_RUNMODE);
    }
    else
    {
        csoca::elog.Print("invalid data type in Grid_FFT<data_t>::setup_fft_interface\n");
    }

    fft_norm_fac_ = 1.f / sqrtf((float)((size_t)n_[0] * (size_t)n_[1] * (size_t)n_[2]));

    if (typeid(data_t) == typeid(real_t))
    {
        npr_ = n_[2] + 2;
        npc_ = n_[2] / 2 + 1;
    }
    else
    {
        npr_ = n_[2];
        npc_ = n_[2];
    }

    nhalf_[0] = n_[0] / 2;
    nhalf_[1] = n_[1] / 2;
    nhalf_[2] = n_[2] / 2;

    kfac_[0] = (real_t)(2.0 * M_PI) / length_[0];
    kfac_[1] = (real_t)(2.0 * M_PI) / length_[1];
    kfac_[2] = (real_t)(2.0 * M_PI) / length_[2];

    local_0_size_ = n_[0];
    local_1_size_ = n_[1];

    if (space_ == rspace_id)
    {
        sizes_[0] = n_[0];
        sizes_[1] = n_[1];
        sizes_[2] = n_[2];
        sizes_[3] = npr_;
    }
    else
    {
        sizes_[0] = n_[1];
        sizes_[1] = n_[0];
        sizes_[2] = npc_;
        sizes_[3] = npc_;
    }

    global_range_.x1_[0] = 0;
    global_range_.x1_[1] = 0;
    global_range_.x1_[2] = 0;

    global_range_.x2_[0] = n_[0];
    global_range_.x2_[1] = n_[1];
    global_range_.x2_[2] = n_[2];

#else //// i.e. ifdef USE_MPI ////////////////////////////////////////////////////////////////////////////////////

    size_t cmplxsz;

    if (typeid(data_t) == typeid(real_t))
    {
        cmplxsz = FFTW_API(mpi_local_size_3d_transposed)(n_[0], n_[1], n_[2] / 2 + 1, MPI_COMM_WORLD,
                                                         &local_0_size_, &local_0_start_, &local_1_size_, &local_1_start_);
        ntot_ = 2 * cmplxsz;
        data_ = (data_t*)fftw_malloc(ntot_ * sizeof(real_t));
        cdata_ = reinterpret_cast<ccomplex_t *>(data_);
        plan_ = FFTW_API(mpi_plan_dft_r2c_3d)(n_[0], n_[1], n_[2], (real_t *)data_, (complex_t *)data_,
                                              MPI_COMM_WORLD, FFTW_RUNMODE | FFTW_MPI_TRANSPOSED_OUT);
        iplan_ = FFTW_API(mpi_plan_dft_c2r_3d)(n_[0], n_[1], n_[2], (complex_t *)data_, (real_t *)data_,
                                               MPI_COMM_WORLD, FFTW_RUNMODE | FFTW_MPI_TRANSPOSED_IN);
    }
    else if (typeid(data_t) == typeid(ccomplex_t))
    {
        cmplxsz = FFTW_API(mpi_local_size_3d_transposed)(n_[0], n_[1], n_[2], MPI_COMM_WORLD,
                                                         &local_0_size_, &local_0_start_, &local_1_size_, &local_1_start_);
        ntot_ = cmplxsz;
        data_ = (data_t*)fftw_malloc(ntot_ * sizeof(ccomplex_t));
        cdata_ = reinterpret_cast<ccomplex_t *>(data_);
        plan_ = FFTW_API(mpi_plan_dft_3d)(n_[0], n_[1], n_[2], (complex_t *)data_, (complex_t *)data_,
                                            MPI_COMM_WORLD, FFTW_FORWARD, FFTW_RUNMODE | FFTW_MPI_TRANSPOSED_OUT);
        iplan_ = FFTW_API(mpi_plan_dft_3d)(n_[0], n_[1], n_[2], (complex_t *)data_, (complex_t *)data_,
                                           MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_RUNMODE | FFTW_MPI_TRANSPOSED_IN);
    }
    else
    {
        csoca::elog.Print("unknown data type in Grid_FFT<data_t>::setup_fft_interface\n");
        abort();
    }

    
    csoca::ilog.Print("[FFT] Setting up a distributed memory field %lux%lux%lu\n", n_[0], n_[1], n_[2]);


    fft_norm_fac_ = 1.0 / sqrt((double)n_[0] * (double)n_[1] * (double)n_[2]);

    if (typeid(data_t) == typeid(real_t))
    {
        npr_ = n_[2] + 2;
        npc_ = n_[2] / 2 + 1;
    }
    else
    {
        npr_ = n_[2];
        npc_ = n_[2];
    }

    nhalf_[0] = n_[0] / 2;
    nhalf_[1] = n_[1] / 2;
    nhalf_[2] = n_[2] / 2;

    kfac_[0] = (real_t)(2.0 * M_PI) / length_[0];
    kfac_[1] = (real_t)(2.0 * M_PI) / length_[1];
    kfac_[2] = (real_t)(2.0 * M_PI) / length_[2];

    if (space_ == rspace_id)
    {
        sizes_[0] = (int)local_0_size_;
        sizes_[1] = n_[1];
        sizes_[2] = n_[2];
        sizes_[3] = npr_; // holds the physical memory size along the 3rd dimension
    }
    else
    {
        sizes_[0] = (int)local_1_size_;
        sizes_[1] = n_[0];
        sizes_[2] = npc_;
        sizes_[3] = npc_; // holds the physical memory size along the 3rd dimension
    }

    global_range_.x1_[0] = (int)local_0_start_;
    global_range_.x1_[1] = 0;
    global_range_.x1_[2] = 0;

    global_range_.x2_[0] = (int)(local_0_start_ + local_0_size_);
    global_range_.x2_[1] = n_[1];
    global_range_.x2_[2] = n_[2];

#endif //// of #ifdef #else USE_MPI ////////////////////////////////////////////////////////////////////////////////////
}

template <typename data_t>
void Grid_FFT<data_t>::ApplyNorm(void)
{
    #pragma omp parallel for
    for (size_t i = 0; i < ntot_; ++i)
        data_[i] *= fft_norm_fac_;
}

template <typename data_t>
void Grid_FFT<data_t>::FourierTransformForward(bool do_transform)
{

    if (space_ != kspace_id)
    {
        //.............................
        if (do_transform)
        {
            double wtime = get_wtime();

            FFTW_API(execute)
            (plan_);
            this->ApplyNorm();

            wtime = get_wtime() - wtime;
            csoca::ilog.Print("[FFT] Completed Grid_FFT::to_kspace (%lux%lux%lu), took %f s", sizes_[0], sizes_[1], sizes_[2], wtime);
        }

        sizes_[0] = local_1_size_;
        sizes_[1] = n_[0];
        sizes_[2] = (int)npc_;
        sizes_[3] = npc_;

        space_ = kspace_id;
        //.............................
    }
}

template <typename data_t>
void Grid_FFT<data_t>::FourierTransformBackward(bool do_transform)
{
    if (space_ != rspace_id)
    {
        //.............................
        if (do_transform)
        {
            double wtime = get_wtime();

            FFTW_API(execute)
            (iplan_);
            this->ApplyNorm();

            wtime = get_wtime() - wtime;
            csoca::ilog.Print("[FFT] Completed Grid_FFT::to_rspace (%dx%dx%d), took %f s\n", sizes_[0], sizes_[1], sizes_[2], wtime);
        }
        sizes_[0] = local_0_size_;
        sizes_[1] = n_[1];
        sizes_[2] = n_[2];
        sizes_[3] = npr_;

        space_ = rspace_id;
        //.............................
    }
}

#define H5_USE_16_API
#include <hdf5.h>

bool file_exists(std::string Filename)
{
    bool flag = false;
    std::fstream fin(Filename.c_str(), std::ios::in | std::ios::binary);
    if (fin.is_open())
        flag = true;
    fin.close();
    return flag;
}

void create_hdf5(std::string Filename)
{
    hid_t HDF_FileID;
    HDF_FileID =
        H5Fcreate(Filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    H5Fclose(HDF_FileID);
}

template <typename data_t>
void Grid_FFT<data_t>::Write_to_HDF5(std::string fname, std::string datasetname)
{
    const bool bComplexType(typeid(data_t) == typeid(std::complex<double>) ||
                            typeid(data_t) == typeid(std::complex<float>));

    std::string datasetname_real = datasetname + std::string("_real");
    std::string datasetname_imag = datasetname + std::string("_imag");

    hid_t file_id, dset_id;    /* file and dataset identifiers */
    hid_t filespace, memspace; /* file and memory dataspace identifiers */
    hsize_t offset[3], count[3];
    hid_t dtype_id;
    hid_t plist_id;

#if defined(USE_MPI)

    if (!file_exists(fname) && CONFIG::MPI_task_rank == 0)
        create_hdf5(fname);
    MPI_Barrier(MPI_COMM_WORLD);

#ifndef NOMPIIO
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
#else
    plist_id = H5P_DEFAULT;
#endif

#else

    if (!file_exists(fname))
        create_hdf5(fname);

    plist_id = H5P_DEFAULT;

#endif

#if defined(USE_MPI) && defined(NOMPIIO)
    for (int itask = 0; itask < CONFIG::MPI_task_size; ++itask)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (itask != CONFIG::MPI_task_rank)
            continue;

#endif

        // file_id = H5Fcreate( fname.c_str(), H5F_ACC_RDWR, H5P_DEFAULT,
        // H5P_DEFAULT );
        file_id = H5Fopen(fname.c_str(), H5F_ACC_RDWR, plist_id);

        for (int i = 0; i < 3; ++i)
            count[i] = size(i);

        assert(typeid(real_t) == typeid(float) || typeid(real_t) == typeid(double));

        if (typeid(real_t) == typeid(float))
            dtype_id = H5T_NATIVE_FLOAT;
        else
            dtype_id = H5T_NATIVE_DOUBLE;

        hsize_t slice_sz = size(1) * size(2);

        count[0] = size(0);
        count[1] = size(1);
        count[2] = size(2);

        offset[1] = 0;
        offset[2] = 0;

#if defined(USE_MPI) && !defined(NOMPIIO)
        H5Pclose(plist_id);
        plist_id = H5Pcreate(H5P_DATASET_XFER);
// H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#else
    plist_id = H5P_DEFAULT;
#endif

        real_t *buf = new real_t[slice_sz];

        //-------- write real part
        //-----------------------------------------------------------

#if defined(USE_MPI) && defined(NOMPIIO)
        if (itask == 0)
        {
            filespace = H5Screate_simple(3, count, NULL);
            dset_id = H5Dcreate2(file_id, datasetname.c_str(), dtype_id, filespace,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(filespace);
        }
        else
        {
            dset_id = H5Dopen1(file_id, datasetname.c_str());
        }
#else
    filespace = H5Screate_simple(3, count, NULL);
    dset_id = H5Dcreate2(file_id, datasetname.c_str(), dtype_id, filespace,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);
#endif

        count[0] = 1;

        for (size_t i = 0; i < size(0); ++i)
        {
            offset[0] = i;

            for (size_t j = 0; j < size(1); ++j)
                for (size_t k = 0; k < size(2); ++k){
                    buf[j * size(2) + k] = std::real(relem(i, j, k));
                }

            memspace = H5Screate_simple(3, count, NULL);

            filespace = H5Dget_space(dset_id);
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
            // H5Dwrite( dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT,
            // m_Data );

            H5Dwrite(dset_id, dtype_id, memspace, filespace, plist_id, buf);
            H5Sclose(memspace);
            H5Sclose(filespace);
        }

        H5Dclose(dset_id);

        //-------- write imaginary part
        //-----------------------------------------------------------
        if( bComplexType ){

            count[0] = size(0);

#if defined(USE_MPI) && defined(NOMPIIO)
            if (itask == 0)
            {
                filespace = H5Screate_simple(3, count, NULL);
                dset_id = H5Dcreate2(file_id, datasetname_imag.c_str(), dtype_id,
                                     filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                H5Sclose(filespace);
            }
            else
            {
                dset_id = H5Dopen1(file_id, datasetname_imag.c_str());
            }
#else
            filespace = H5Screate_simple(3, count, NULL);
            dset_id = H5Dcreate2(file_id, datasetname_imag.c_str(), dtype_id, filespace,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(filespace);
#endif
            count[0] = 1;

            for (size_t i = 0; i < size(0); ++i)
            {
                offset[0] = i;

                for (size_t j = 0; j < size(1); ++j)
                    for (size_t k = 0; k < size(2); ++k)
                        buf[j * size(2) + k] = std::imag(relem(i, j, k));

                memspace = H5Screate_simple(3, count, NULL);

                filespace = H5Dget_space(dset_id);
                H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
                // H5Dwrite( dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT,
                // m_Data );
                H5Dwrite(dset_id, dtype_id, memspace, filespace, H5P_DEFAULT, buf);
                H5Sclose(memspace);
                H5Sclose(filespace);
            }

            H5Dclose(dset_id);
        }
        //------------------------------------------------------------------------------------

#if defined(USE_MPI) && !defined(NOMPIIO)
        H5Pclose(plist_id);
#endif

        H5Fclose(file_id);

        delete[] buf;

#if defined(USE_MPI) && defined(NOMPIIO)
    }
#endif
}

#include <iomanip>

template <typename data_t>
void Grid_FFT<data_t>::Compute_PDF( std::string ofname, int nbins, double scale, double vmin, double vmax )
{
    double logvmin = std::log10(vmin);
    double logvmax = std::log10(vmax);
    double idv = double(nbins) / (logvmax - logvmin);

    std::vector<double> count( nbins, 0.0 ), scount( nbins, 0.0 );

    for( size_t ix=0; ix<size(0); ix++ )
        for( size_t iy=0; iy<size(1); iy++ )
            for (size_t iz = 0; iz < size(2); iz++)
            {
                auto v = this->relem(ix,iy,iz);
                int ibin = int( (std::log10(std::abs(v))-logvmin)*idv );
                if( ibin >= 0 && ibin < nbins ){
                    count[ibin] += 1.0;
                }
                ibin = int(((std::log10((std::abs(v)-1.0) * scale + 1.0 )) - logvmin) * idv);
                if (ibin >= 0 && ibin < nbins)
                {
                    scount[ibin] += 1.0;
                }
            }

#if defined(USE_MPI)
    if (CONFIG::MPI_task_rank == 0)
    {
#endif
        std::ofstream ofs(ofname.c_str());
        std::size_t numcells = size(0) * size(1) * size(2);

        //ofs << "# a = " << aexp << std::endl;
        ofs << "# " << std::setw(14) << "rho" << std::setw(16) << "d rho / dV" << std::setw(16) << "d (rho/D+) / dV"
            << "\n";

        for( int ibin=0; ibin<nbins; ++ibin ){
            double vmean = std::pow(10.0, logvmin + (double(ibin)+0.5)/idv );
            double dv = std::pow(10.0, logvmin + (double(ibin) + 1.0) / idv) - std::pow(10.0, logvmin + (double(ibin)) / idv);

            ofs << std::setw(16) << vmean
                << std::setw(16) << count[ibin] / dv / numcells
                << std::setw(16) << scount[ibin] / dv / numcells
                << std::endl;
        }
#if defined(USE_MPI)
    }
#endif
}

template <typename data_t>
void Grid_FFT<data_t>::ComputePowerSpectrum(std::vector<double> &bin_k, std::vector<double> &bin_P, std::vector<double> &bin_eP, int nbins)
{
    real_t kmax = std::max(std::max(kfac_[0] * nhalf_[0], kfac_[1] * nhalf_[1]),
                           kfac_[2] * nhalf_[2]),
           kmin = std::min(std::min(kfac_[0], kfac_[1]), kfac_[2]),
           dklog = log10(kmax / kmin) / nbins;

    std::vector<size_t> bin_count;

    bin_count.assign(nbins,0);
    bin_k.assign(nbins, 0);
    bin_P.assign(nbins, 0);
    bin_eP.assign(nbins, 0);

    for (size_t ix = 0; ix < size(0); ix++)
        for (size_t iy = 0; iy < size(1); iy++)
            for (size_t iz = 0; iz < size(2); iz++)
            {
                vec3<double> k3 = get_k<double>(ix, iy, iz);
                double k = k3.norm();
                int idx2 = int((1.0f / dklog * std::log10(k / kmin)));
                auto z = this->kelem(ix, iy, iz);
                double vabs = z.real()*z.real()+z.imag()*z.imag();

                if (k >= kmin && k < kmax)
                {
                    if (iz == 0)
                    {
                        bin_P[idx2] += vabs;
                        bin_eP[idx2] += vabs * vabs;
                        bin_k[idx2] += k;
                        bin_count[idx2]++;
                    }
                    else
                    {
                        bin_P[idx2] += 2.0 * vabs;
                        bin_eP[idx2] += 2.0 * vabs * vabs;
                        bin_k[idx2] += 2.0 * k;
                        bin_count[idx2] += 2;
                    }
                }
            }

#if defined(USE_MPI)
    std::vector<double> tempv(nbins, 0);
    std::vector<size_t> tempvi(nbins, 0);

    MPI_Allreduce(reinterpret_cast<void *>(&bin_k[0]), reinterpret_cast<void *>(&tempv[0]),
                  nbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    bin_k.swap(tempv);
    MPI_Allreduce(reinterpret_cast<void *>(&bin_P[0]), reinterpret_cast<void *>(&tempv[0]),
                  nbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    bin_P.swap(tempv);
    MPI_Allreduce(reinterpret_cast<void *>(&bin_eP[0]), reinterpret_cast<void *>(&tempv[0]),
                  nbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    bin_eP.swap(tempv);
    MPI_Allreduce(reinterpret_cast<void *>(&bin_count[0]), reinterpret_cast<void *>(&tempvi[0]),
                  nbins, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    bin_count.swap(tempvi);

#endif

    const real_t volfac(length_[0] * length_[1] * length_[2] / std::pow(2.0 * M_PI, 3.0));
    const real_t fftfac(std::pow(double(size(0)), 3.0));

    for( int i=0; i<nbins; ++i ){
        bin_k[i]  /= bin_count[i];
        bin_P[i]  = bin_P[i] / bin_count[i] * volfac / fftfac;
        bin_eP[i] = std::sqrt(bin_eP[i] / bin_count[i] - bin_P[i] * bin_P[i])/std::sqrt(bin_count[i]) * volfac / fftfac;
    }
}

template <typename array_type>
int get_task(ptrdiff_t index, const array_type &offsets, const array_type& sizes,
             const int ntasks )
{
    int itask = 0;
    while (itask < ntasks - 1 && offsets[itask + 1] <= index)
        ++itask;
    return itask;
}

template <typename data_t>
void pad_insert(const Grid_FFT<data_t> &f, Grid_FFT<data_t> &fp)
{

    // assert(fp.n_[0] == 3 * f.n_[0] / 2);
    // assert(fp.n_[1] == 3 * f.n_[1] / 2);
    // assert(fp.n_[2] == 3 * f.n_[2] / 2);

    size_t dn[3] = {
        fp.n_[0] - f.n_[0],
        fp.n_[1] - f.n_[1],
        fp.n_[2] - f.n_[2],
    };
    const double rfac = std::sqrt(fp.n_[0] * fp.n_[1] * fp.n_[2]) / std::sqrt(f.n_[0] * f.n_[1] * f.n_[2]);

    fp.zero();

#if !defined(USE_MPI) ////////////////////////////////////////////////////////////////////////////////////

    size_t nhalf[3] = {f.n_[0] / 2, f.n_[1] / 2, f.n_[2] / 2};

    for (size_t i = 0; i < f.size(0); ++i)
    {
        size_t ip = (i > nhalf[0]) ? i + dn[0] : i;
        for (size_t j = 0; j < f.size(1); ++j)
        {
            size_t jp = (j > nhalf[1]) ? j + dn[1] : j;
            for (size_t k = 0; k < f.size(2); ++k)
            {
                size_t kp = (k > nhalf[2]) ? k + dn[2] : k;
                // if( i==nhalf[0]||j==nhalf[1]||k==nhalf[2]) continue;
                fp.kelem(ip, jp, kp) = f.kelem(i, j, k) * rfac;
            }
        }
    }


#else /// then USE_MPI is defined ////////////////////////////////////////////////////////////

    MPI_Barrier(MPI_COMM_WORLD);

    /////////////////////////////////////////////////////////////////////
    size_t maxslicesz = fp.sizes_[1] * fp.sizes_[3] * 2;

    std::vector<ccomplex_t> crecvbuf_(maxslicesz / 2, 0);
    real_t *recvbuf_ = reinterpret_cast<real_t *>(&crecvbuf_[0]);

    std::vector<ptrdiff_t>
        offsets_(CONFIG::MPI_task_size, 0),
        offsetsp_(CONFIG::MPI_task_size, 0),
        sizes_(CONFIG::MPI_task_size, 0),
        sizesp_(CONFIG::MPI_task_size, 0);

    size_t tsize = f.size(0), tsizep = fp.size(0);

    MPI_Allgather(&f.local_1_start_, 1, MPI_LONG_LONG, &offsets_[0], 1,
                  MPI_LONG_LONG, MPI_COMM_WORLD);
    MPI_Allgather(&fp.local_1_start_, 1, MPI_LONG_LONG, &offsetsp_[0], 1,
                  MPI_LONG_LONG, MPI_COMM_WORLD);
    MPI_Allgather(&tsize, 1, MPI_LONG_LONG, &sizes_[0], 1, MPI_LONG_LONG,
                  MPI_COMM_WORLD);
    MPI_Allgather(&tsizep, 1, MPI_LONG_LONG, &sizesp_[0], 1, MPI_LONG_LONG,
                  MPI_COMM_WORLD);
    /////////////////////////////////////////////////////////////////////

    double tstart = get_wtime();
    csoca::dlog << "[MPI] Started scatter for convolution" << std::endl;

    //... collect offsets

    assert(f.space_ == kspace_id);

    size_t nf[3] = {f.size(0), f.size(1), f.size(2)};
    size_t nfp[3] = {fp.size(0), fp.size(1), fp.size(2)};
    size_t fny[3] = {f.n_[1] / 2, f.n_[0] / 2, f.n_[2] / 2};

    //... local size must be divisible by 2, otherwise this gets too complicated
    assert(f.n_[1] % 2 == 0);

    size_t slicesz = f.size(1) * f.size(3); //*2;

    // comunicate
    if (typeid(data_t) == typeid(real_t))
        slicesz *= 2; // then sizeof(real_t) gives only half of a complex

    MPI_Datatype datatype =
        (typeid(data_t) == typeid(float))
            ? MPI_FLOAT
            : (typeid(data_t) == typeid(double))
                  ? MPI_DOUBLE
                  : (typeid(data_t) == typeid(std::complex<float>))
                        ? MPI_COMPLEX
                        : (typeid(data_t) == typeid(std::complex<double>))
                              ? MPI_DOUBLE_COMPLEX
                              : MPI_INT;

    MPI_Status status;

    std::vector<MPI_Request> req;
    MPI_Request temp_req;

    for (size_t i = 0; i < nf[0]; ++i)
    {
        size_t iglobal = i + offsets_[CONFIG::MPI_task_rank];

        if (iglobal < fny[0])
        {
            int sendto = get_task(iglobal, offsetsp_, sizesp_, CONFIG::MPI_task_size);
            MPI_Isend(&f.kelem(i * slicesz), (int)slicesz, datatype, sendto,
                      (int)iglobal, MPI_COMM_WORLD, &temp_req);
            req.push_back(temp_req);
            // ofs << "task " << CONFIG::MPI_task_rank << " : added request No" << req.size()-1 << ":
            // Isend #" << iglobal << " to task " << sendto << std::endl;
        }
        if (iglobal > fny[0])
        {
            int sendto = get_task(iglobal + fny[0], offsetsp_, sizesp_, CONFIG::MPI_task_size);
            MPI_Isend(&f.kelem(i * slicesz), (int)slicesz, datatype, sendto,
                      (int)(iglobal + fny[0]), MPI_COMM_WORLD, &temp_req);
            req.push_back(temp_req);
            // ofs << "task " << CONFIG::MPI_task_rank << " : added request No" << req.size()-1 << ":
            // Isend #" << iglobal+fny[0] << " to task " << sendto << std::endl;
        }
    }

    for (size_t i = 0; i < nfp[0]; ++i)
    {
        size_t iglobal = i + offsetsp_[CONFIG::MPI_task_rank];

        if (iglobal < fny[0] || iglobal > 2 * fny[0])
        {
            int recvfrom = 0;
            if (iglobal <= fny[0])
                recvfrom = get_task(iglobal, offsets_, sizes_, CONFIG::MPI_task_size);
            else
                recvfrom = get_task(iglobal - fny[0], offsets_, sizes_, CONFIG::MPI_task_size);

            // ofs << "task " << CONFIG::MPI_task_rank << " : receive #" << iglobal << " from task "
            // << recvfrom << std::endl;

            MPI_Recv(&recvbuf_[0], (int)slicesz, datatype, recvfrom, (int)iglobal,
                     MPI_COMM_WORLD, &status);
            // ofs << "---> ok!  " << (bool)(status.Get_error()==MPI::SUCCESS) <<
            // std::endl;

            assert(status.MPI_ERROR == MPI_SUCCESS);

            for (size_t j = 0; j < nf[1]; ++j)
            {
                if (j < fny[1])
                {
                    size_t jp = j;
                    for (size_t k = 0; k < nf[2]; ++k)
                    {
                        // size_t kp = (k>fny[2])? k+fny[2] : k;
                        if (k < fny[2])
                            fp.kelem(i, jp, k) = crecvbuf_[j * f.sizes_[3] + k];
                        else if (k > fny[2])
                            fp.kelem(i, jp, k + fny[2]) = crecvbuf_[j * f.sizes_[3] + k];
                    }
                }

                else if (j > fny[1])
                {
                    size_t jp = j + fny[1];
                    for (size_t k = 0; k < nf[2]; ++k)
                    {
                        // size_t kp = (k>fny[2])? k+fny[2] : k;
                        // fp.kelem(i,jp,kp) = crecvbuf_[j*f.sizes_[3]+k];
                        if (k < fny[2])
                            fp.kelem(i, jp, k) = crecvbuf_[j * f.sizes_[3] + k];
                        else if (k > fny[2])
                            fp.kelem(i, jp, k + fny[2]) = crecvbuf_[j * f.sizes_[3] + k];
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < req.size(); ++i)
    {
        // need to set status as wait does not necessarily modify it
        // c.f. http://www.open-mpi.org/community/lists/devel/2007/04/1402.php
        status.MPI_ERROR = MPI_SUCCESS;
        // ofs << "task " << CONFIG::MPI_task_rank << " : checking request No" << i << std::endl;
        MPI_Wait(&req[i], &status);
        // ofs << "---> ok!" << std::endl;
        assert(status.MPI_ERROR == MPI_SUCCESS);
    }

    // usleep(1000);

    MPI_Barrier(MPI_COMM_WORLD);

    // std::cerr << ">>>>> task " << CONFIG::MPI_task_rank << " all transfers completed! <<<<<"
    // << std::endl;  ofs << ">>>>> task " << CONFIG::MPI_task_rank << " all transfers completed!
    // <<<<<" << std::endl;
    csoca::dlog.Print("[MPI] Completed scatter for convolution, took %fs\n",
                get_wtime() - tstart);

#endif /// end of ifdef/ifndef USE_MPI ///////////////////////////////////////////////////////////////
}


template <typename data_t>
void unpad(const Grid_FFT<data_t> &fp, Grid_FFT<data_t> &f)
{
    // assert(fp.n_[0] == 3 * f.n_[0] / 2);
    // assert(fp.n_[1] == 3 * f.n_[1] / 2);
    // assert(fp.n_[2] == 3 * f.n_[2] / 2);

    size_t dn[3] = {
        fp.n_[0] - f.n_[0],
        fp.n_[1] - f.n_[1],
        fp.n_[2] - f.n_[2],
    };

    const double rfac = std::sqrt(fp.n_[0] * fp.n_[1] * fp.n_[2]) / std::sqrt(f.n_[0] * f.n_[1] * f.n_[2]);

#if !defined(USE_MPI) ////////////////////////////////////////////////////////////////////////////////////

    size_t nhalf[3] = {f.n_[0] / 2, f.n_[1] / 2, f.n_[2] / 2};

    for (size_t i = 0; i < f.size(0); ++i)
    {
        size_t ip = (i > nhalf[0]) ? i + dn[0] : i;
        for (size_t j = 0; j < f.size(1); ++j)
        {
            size_t jp = (j > nhalf[1]) ? j + dn[1] : j;
            for (size_t k = 0; k < f.size(2); ++k)
            {
                size_t kp = (k > nhalf[2]) ? k + dn[2] : k;
                // if( i==nhalf[0]||j==nhalf[1]||k==nhalf[2]) continue;
                f.kelem(i, j, k) = fp.kelem(ip, jp, kp) / rfac;
            }
        }
    }

#else /// then USE_MPI is defined //////////////////////////////////////////////////////////////

                        /////////////////////////////////////////////////////////////////////
                        size_t maxslicesz = fp.sizes_[1] * fp.sizes_[3] * 2;

    std::vector<ccomplex_t> crecvbuf_(maxslicesz / 2,0);
    real_t* recvbuf_ = reinterpret_cast<real_t *>(&crecvbuf_[0]);

    std::vector<ptrdiff_t> 
        offsets_(CONFIG::MPI_task_size, 0), 
        offsetsp_(CONFIG::MPI_task_size, 0), 
        sizes_(CONFIG::MPI_task_size, 0), 
        sizesp_(CONFIG::MPI_task_size, 0);

    size_t tsize = f.size(0), tsizep = fp.size(0);

    MPI_Allgather(&f.local_1_start_, 1, MPI_LONG_LONG, &offsets_[0], 1,
                  MPI_LONG_LONG, MPI_COMM_WORLD);
    MPI_Allgather(&fp.local_1_start_, 1, MPI_LONG_LONG, &offsetsp_[0], 1,
                  MPI_LONG_LONG, MPI_COMM_WORLD);
    MPI_Allgather(&tsize, 1, MPI_LONG_LONG, &sizes_[0], 1, MPI_LONG_LONG,
                  MPI_COMM_WORLD);
    MPI_Allgather(&tsizep, 1, MPI_LONG_LONG, &sizesp_[0], 1, MPI_LONG_LONG,
                  MPI_COMM_WORLD);
    /////////////////////////////////////////////////////////////////////

    double tstart = get_wtime();

    csoca::ilog << "[MPI] Started gather for convolution";

    MPI_Barrier(MPI_COMM_WORLD);

    size_t nf[3] = {f.size(0), f.size(1), f.size(2)};
    size_t nfp[4] = {fp.size(0), fp.size(1), fp.size(2), fp.size(3)};
    size_t fny[3] = {f.n_[1] / 2, f.n_[0] / 2, f.n_[2] / 2};

    size_t slicesz = fp.size(1) * fp.size(3);

    if (typeid(data_t) == typeid(real_t))
        slicesz *= 2; // then sizeof(real_t) gives only half of a complex

    MPI_Datatype datatype =
        (typeid(data_t) == typeid(float))
            ? MPI_FLOAT
            : (typeid(data_t) == typeid(double))
                  ? MPI_DOUBLE
                  : (typeid(data_t) == typeid(std::complex<float>))
                        ? MPI_COMPLEX
                        : (typeid(data_t) == typeid(std::complex<double>))
                              ? MPI_DOUBLE_COMPLEX
                              : MPI_INT;

    MPI_Status status;

    //... local size must be divisible by 2, otherwise this gets too complicated
    // assert( tsize%2 == 0 );

    f.zero();

    std::vector<MPI_Request> req;
    MPI_Request temp_req;

    for (size_t i = 0; i < nfp[0]; ++i)
    {
        size_t iglobal = i + offsetsp_[CONFIG::MPI_task_rank];

        //... sending
        if (iglobal < fny[0])
        {
            int sendto = get_task(iglobal, offsets_, sizes_, CONFIG::MPI_task_size);

            MPI_Isend(&fp.kelem(i * slicesz), (int)slicesz, datatype, sendto, (int)iglobal,
                      MPI_COMM_WORLD, &temp_req);
            req.push_back(temp_req);
        }
        else if (iglobal > 2 * fny[0])
        {
            int sendto = get_task(iglobal - fny[0], offsets_, sizes_, CONFIG::MPI_task_size);
            MPI_Isend(&fp.kelem(i * slicesz), (int)slicesz, datatype, sendto, (int)iglobal,
                      MPI_COMM_WORLD, &temp_req);
            req.push_back(temp_req);
        }
    }

    for (size_t i = 0; i < nf[0]; ++i)
    {
        size_t iglobal = i + offsets_[CONFIG::MPI_task_rank];

        int recvfrom = 0;
        if (iglobal < fny[0])
        {
            recvfrom = get_task(iglobal, offsetsp_, sizesp_, CONFIG::MPI_task_size);
            MPI_Recv(&recvbuf_[0], (int)slicesz, datatype, recvfrom, (int)iglobal,
                     MPI_COMM_WORLD, &status);
        }
        else if (iglobal > fny[0])
        {
            recvfrom = get_task(iglobal + fny[0], offsetsp_, sizesp_, CONFIG::MPI_task_size);
            MPI_Recv(&recvbuf_[0], (int)slicesz, datatype, recvfrom,
                     (int)(iglobal + fny[0]), MPI_COMM_WORLD, &status);
        }
        else
            continue;

        assert(status.MPI_ERROR == MPI_SUCCESS);

        for (size_t j = 0; j < nf[1]; ++j)
        {

            if (j < fny[1])
            {
                size_t jp = j;
                for (size_t k = 0; k < nf[2]; ++k)
                {
                    // size_t kp = (k>fny[2])? k+fny[2] : k;
                    // f.kelem(i,j,k) = crecvbuf_[jp*nfp[3]+kp];
                    if (k < fny[2])
                        f.kelem(i, j, k) = crecvbuf_[jp * nfp[3] + k];
                    else if (k > fny[2])
                        f.kelem(i, j, k) = crecvbuf_[jp * nfp[3] + k + fny[2]];
                }
            }
            if (j > fny[1])
            {
                size_t jp = j + fny[1];
                for (size_t k = 0; k < nf[2]; ++k)
                {
                    // size_t kp = (k>fny[2])? k+fny[2] : k;
                    // f.kelem(i,j,k) = crecvbuf_[jp*nfp[3]+kp];
                    if (k < fny[2])
                        f.kelem(i, j, k) = crecvbuf_[jp * nfp[3] + k];
                    else if (k > fny[2])
                        f.kelem(i, j, k) = crecvbuf_[jp * nfp[3] + k + fny[2]];
                }
            }
        }
    }

    for (size_t i = 0; i < req.size(); ++i)
    {
        // need to preset status as wait does not necessarily modify it to reflect
        // success c.f.
        // http://www.open-mpi.org/community/lists/devel/2007/04/1402.php
        status.MPI_ERROR = MPI_SUCCESS;

        MPI_Wait(&req[i], &status);
        assert(status.MPI_ERROR == MPI_SUCCESS);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    csoca::ilog.Print("[MPI] Completed gather for convolution, took %fs", get_wtime() - tstart);

#endif /// end of ifdef/ifndef USE_MPI //////////////////////////////////////////////////////////////
}

/********************************************************************************************/


template class Grid_FFT<real_t>;
template class Grid_FFT<ccomplex_t>;

template void unpad(const Grid_FFT<real_t> &fp, Grid_FFT<real_t> &f);
template void unpad(const Grid_FFT<ccomplex_t> &fp, Grid_FFT<ccomplex_t> &f);

template void pad_insert(const Grid_FFT<real_t> &f, Grid_FFT<real_t> &fp);
template void pad_insert(const Grid_FFT<ccomplex_t> &f, Grid_FFT<ccomplex_t> &fp);
