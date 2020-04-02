#include <general.hh>
#include <grid_fft.hh>
#include <thread>

template <typename data_t,bool bdistributed>
void Grid_FFT<data_t,bdistributed>::Setup(void)
{
    if( !bdistributed ){
        ntot_ = (n_[2] + 2) * n_[1] * n_[0];

        csoca::dlog.Print("[FFT] Setting up a shared memory field %lux%lux%lu\n", n_[0], n_[1], n_[2]);
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

        fft_norm_fac_ = 1.0 / std::sqrt((double)((size_t)n_[0] * (double)n_[1] * (double)n_[2]));

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

        for (int i = 0; i < 3; ++i)
        {
            nhalf_[i] = n_[i] / 2;
            kfac_[i] = 2.0 * M_PI / length_[i];
            dx_[i] = length_[i] / n_[i];

            global_range_.x1_[i] = 0;
            global_range_.x2_[i] = n_[i];
        }

        local_0_size_ = n_[0];
        local_1_size_ = n_[1];
        local_0_start_ = 0;
        local_1_start_ = 0;

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
    }
    else
    {
#ifdef USE_MPI //// i.e. ifdef USE_MPI ////////////////////////////////////////////////////////////////////////////////////
        size_t cmplxsz;

        if (typeid(data_t) == typeid(real_t))
        {
            cmplxsz = FFTW_API(mpi_local_size_3d_transposed)(n_[0], n_[1], n_[2] / 2 + 1, MPI_COMM_WORLD,
                                                            &local_0_size_, &local_0_start_, &local_1_size_, &local_1_start_);
            ntot_ = 2 * cmplxsz;
            data_ = (data_t *)fftw_malloc(ntot_ * sizeof(real_t));
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
            data_ = (data_t *)fftw_malloc(ntot_ * sizeof(ccomplex_t));
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

        csoca::dlog.Print("[FFT] Setting up a distributed memory field %lux%lux%lu\n", n_[0], n_[1], n_[2]);
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

        for (int i = 0; i < 3; ++i)
        {
            nhalf_[i] = n_[i] / 2;
            kfac_[i] = 2.0 * M_PI / length_[i];
            dx_[i] = length_[i] / n_[i];

            global_range_.x1_[i] = 0;
            global_range_.x2_[i] = n_[i];
        }
        global_range_.x1_[0] = (int)local_0_start_;
        global_range_.x2_[0] = (int)(local_0_start_ + local_0_size_);

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
#else
        csoca::flog << "MPI is required for distributed FFT arrays!" << std::endl;
        throw std::runtime_error("MPI is required for distributed FFT arrays!");
#endif //// of #ifdef #else USE_MPI ////////////////////////////////////////////////////////////////////////////////////
    }
}

template <typename data_t,bool bdistributed>
void Grid_FFT<data_t,bdistributed>::ApplyNorm(void)
{
#pragma omp parallel for
    for (size_t i = 0; i < ntot_; ++i)
        data_[i] *= fft_norm_fac_;
}

template <typename data_t,bool bdistributed>
void Grid_FFT<data_t,bdistributed>::FourierTransformForward(bool do_transform)
{
#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (space_ != kspace_id)
    {
        //.............................
        if (do_transform)
        {
            double wtime = get_wtime();
            csoca::dlog.Print("[FFT] Calling Grid_FFT::to_kspace (%lux%lux%lu)", sizes_[0], sizes_[1], sizes_[2]);
            FFTW_API(execute)
            (plan_);
            this->ApplyNorm();

            wtime = get_wtime() - wtime;
            csoca::dlog.Print("[FFT] Completed Grid_FFT::to_kspace (%lux%lux%lu), took %f s", sizes_[0], sizes_[1], sizes_[2], wtime);
        }

        sizes_[0] = local_1_size_;
        sizes_[1] = n_[0];
        sizes_[2] = (int)npc_;
        sizes_[3] = npc_;

        space_ = kspace_id;
        //.............................
    }
}

template <typename data_t,bool bdistributed>
void Grid_FFT<data_t,bdistributed>::FourierTransformBackward(bool do_transform)
{
#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (space_ != rspace_id)
    {
        //.............................
        if (do_transform)
        {
            csoca::dlog.Print("[FFT] Calling Grid_FFT::to_rspace (%dx%dx%d)\n", sizes_[0], sizes_[1], sizes_[2]);
            double wtime = get_wtime();

            FFTW_API(execute)
            (iplan_);
            this->ApplyNorm();

            wtime = get_wtime() - wtime;
            csoca::dlog.Print("[FFT] Completed Grid_FFT::to_rspace (%dx%dx%d), took %f s\n", sizes_[0], sizes_[1], sizes_[2], wtime);
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

template <typename T>
hid_t hdf5_get_data_type(void)
{
    if (typeid(T) == typeid(int))
        return H5T_NATIVE_INT;

    if (typeid(T) == typeid(unsigned))
        return H5T_NATIVE_UINT;

    if (typeid(T) == typeid(float))
        return H5T_NATIVE_FLOAT;

    if (typeid(T) == typeid(double))
        return H5T_NATIVE_DOUBLE;

    if (typeid(T) == typeid(long long))
        return H5T_NATIVE_LLONG;

    if (typeid(T) == typeid(unsigned long long))
        return H5T_NATIVE_ULLONG;

    if (typeid(T) == typeid(size_t))
        return H5T_NATIVE_ULLONG;

    std::cerr << " - Error: [HDF_IO] trying to evaluate unsupported type in GetDataType\n\n";
    return -1;
}

template <typename data_t,bool bdistributed>
void Grid_FFT<data_t,bdistributed>::Read_from_HDF5(const std::string Filename, const std::string ObjName)
{
    if( bdistributed ){
        csoca::elog << "Attempt to read from HDF5 into MPI-distributed array. This is not supported yet!" << std::endl;
        abort();
    }

    hid_t HDF_Type = hdf5_get_data_type<data_t>();

    hid_t HDF_FileID = H5Fopen(Filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    //... save old error handler
    herr_t (*old_func)(void *);
    void *old_client_data;

    H5Eget_auto(&old_func, &old_client_data);

    //... turn off error handling by hdf5 library
    H5Eset_auto(NULL, NULL);

    //... probe dataset opening
    hid_t HDF_DatasetID = H5Dopen(HDF_FileID, ObjName.c_str());

    //... restore previous error handler
    H5Eset_auto(old_func, old_client_data);

    //... dataset did not exist or was empty
    if (HDF_DatasetID < 0)
    {
        csoca::elog << "Dataset \'" << ObjName.c_str() << "\' does not exist or is empty." << std::endl;
        H5Fclose(HDF_FileID);
        abort();
    }

    //... get space associated with dataset and its extensions
    hid_t HDF_DataspaceID = H5Dget_space(HDF_DatasetID);

    int ndims = H5Sget_simple_extent_ndims(HDF_DataspaceID);

    hsize_t dimsize[3];

    H5Sget_simple_extent_dims(HDF_DataspaceID, dimsize, NULL);

    hsize_t HDF_StorageSize = 1;
    for (int i = 0; i < ndims; ++i)
        HDF_StorageSize *= dimsize[i];

    //... adjust the array size to hold the data
    std::vector<data_t> Data;
    Data.reserve(HDF_StorageSize);
    Data.assign(HDF_StorageSize, (data_t)0);

    if (Data.capacity() < HDF_StorageSize)
    {
        csoca::elog << "Not enough memory to store all data in HDFReadDataset!" << std::endl;
        H5Sclose(HDF_DataspaceID);
        H5Dclose(HDF_DatasetID);
        H5Fclose(HDF_FileID);
        abort();
    }

    //... read the dataset
    H5Dread(HDF_DatasetID, HDF_Type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Data[0]);

    if (Data.size() != HDF_StorageSize)
    {
        csoca::elog << "Something went wrong while reading!" << std::endl;
        H5Sclose(HDF_DataspaceID);
        H5Dclose(HDF_DatasetID);
        H5Fclose(HDF_FileID);
        abort();
    }

    H5Sclose(HDF_DataspaceID);
    H5Dclose(HDF_DatasetID);
    H5Fclose(HDF_FileID);

    assert( dimsize[0] == dimsize[1] && dimsize[0] == dimsize[2] );
    csoca::ilog << "Read external constraint data of dimensions " << dimsize[0] << "**3." << std::endl;

    for( size_t i=0; i<3; ++i ) this->n_[i] = dimsize[i];
    this->space_ = rspace_id;

    if (data_ != nullptr)
    {
        fftw_free(data_);
    }
    this->Setup();
    

    //... copy data to internal array ...
    double sum1{0.0}, sum2{0.0};
    #pragma omp parallel for reduction(+:sum1,sum2)
    for (size_t i = 0; i < size(0); ++i)
    {
        for (size_t j = 0; j < size(1); ++j)
        {
            for (size_t k = 0; k < size(2); ++k)
            {
                this->relem(i,j,k) = Data[ (i*size(1) + j)*size(2)+k ];
                sum2 += std::real(this->relem(i,j,k)*this->relem(i,j,k));
                sum1 += std::real(this->relem(i,j,k));
            }
        }
    }
    sum1 /= Data.size();
    sum2 /= Data.size();
    auto stdw = std::sqrt(sum2-sum1*sum1);
    csoca::ilog << "Constraint field has <W>=" << sum1 << ", <W^2>-<W>^2=" << stdw << std::endl;

    #pragma omp parallel for reduction(+:sum1,sum2)
    for (size_t i = 0; i < size(0); ++i)
    {
        for (size_t j = 0; j < size(1); ++j)
        {
            for (size_t k = 0; k < size(2); ++k)
            {
                this->relem(i,j,k) /= stdw;
            }
        }
    }
}

template <typename data_t,bool bdistributed>
void Grid_FFT<data_t,bdistributed>::Write_to_HDF5(std::string fname, std::string datasetname) const
{
    // FIXME: cleanup duplicate code in this function!
    if( !bdistributed && CONFIG::MPI_task_rank==0 ){
        
        hid_t file_id, dset_id;    /* file and dataset identifiers */
        hid_t filespace, memspace; /* file and memory dataspace identifiers */
        hsize_t offset[3], count[3];
        hid_t dtype_id = H5T_NATIVE_FLOAT;
        hid_t plist_id = H5P_DEFAULT;

        if (!file_exists(fname))
            create_hdf5(fname);

        file_id = H5Fopen(fname.c_str(), H5F_ACC_RDWR, plist_id);

        for (int i = 0; i < 3; ++i)
            count[i] = size(i);
        
        if (typeid(data_t) == typeid(float))
            dtype_id = H5T_NATIVE_FLOAT;
        else if (typeid(data_t) == typeid(double))
            dtype_id = H5T_NATIVE_DOUBLE;
        else if (typeid(data_t) == typeid(std::complex<float>))
        {
            dtype_id = H5T_NATIVE_FLOAT;
        }
        else if (typeid(data_t) == typeid(std::complex<double>))
        {
            dtype_id = H5T_NATIVE_DOUBLE;
        }

        filespace = H5Screate_simple(3, count, NULL);
        dset_id = H5Dcreate2(file_id, datasetname.c_str(), dtype_id, filespace,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(filespace);

        hsize_t slice_sz = size(1) * size(2);

        real_t *buf = new real_t[slice_sz];

        count[0] = 1;
        count[1] = size(1);
        count[2] = size(2);

        offset[1] = 0;
        offset[2] = 0;

        memspace = H5Screate_simple(3, count, NULL);
        filespace = H5Dget_space(dset_id);

        for (size_t i = 0; i < size(0); ++i)
        {
            offset[0] = i;
            for (size_t j = 0; j < size(1); ++j)
            {
                for (size_t k = 0; k < size(2); ++k)
                {
                    if( this->space_ == rspace_id )
                        buf[j * size(2) + k] = std::real(relem(i, j, k));
                    else
                        buf[j * size(2) + k] = std::real(kelem(i, j, k));
                }
            }

            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
            H5Dwrite(dset_id, dtype_id, memspace, filespace, H5P_DEFAULT, buf);
        }

        H5Sclose(filespace);
        H5Sclose(memspace);

        // H5Sclose(filespace);
        H5Dclose(dset_id);

        if (typeid(data_t) == typeid(std::complex<float>) ||
            typeid(data_t) == typeid(std::complex<double>) ||
            this->space_ == kspace_id )
        {
            datasetname += std::string(".im");

            for (int i = 0; i < 3; ++i)
                count[i] = size(i);

            filespace = H5Screate_simple(3, count, NULL);
            dset_id = H5Dcreate2(file_id, datasetname.c_str(), dtype_id, filespace,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Sclose(filespace);

            count[0] = 1;

            for (size_t i = 0; i < size(0); ++i)
            {
                offset[0] = i;

                for (size_t j = 0; j < size(1); ++j)
                    for (size_t k = 0; k < size(2); ++k)
                    {
                        if( this->space_ == rspace_id )
                            buf[j * size(2) + k] = std::imag(relem(i, j, k));
                        else
                            buf[j * size(2) + k] = std::imag(kelem(i, j, k));
                    }

                memspace = H5Screate_simple(3, count, NULL);
                filespace = H5Dget_space(dset_id);

                H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count,
                                    NULL);

                H5Dwrite(dset_id, dtype_id, memspace, filespace, H5P_DEFAULT, buf);

                H5Sclose(memspace);
                H5Sclose(filespace);
            }

            H5Dclose(dset_id);

            delete[] buf;
        }

        H5Fclose(file_id);
        return;
    }

    if( !bdistributed && CONFIG::MPI_task_rank!=0 ) return;

    hid_t file_id, dset_id;    /* file and dataset identifiers */
    hid_t filespace, memspace; /* file and memory dataspace identifiers */
    hsize_t offset[3], count[3];
    hid_t dtype_id = H5T_NATIVE_FLOAT;
    hid_t plist_id;


#if defined(USE_MPI)

    int mpi_size, mpi_rank;

    mpi_size = MPI::get_size();
    mpi_rank = MPI::get_rank();

    if (!file_exists(fname) && mpi_rank == 0)
        create_hdf5(fname);
    MPI_Barrier(MPI_COMM_WORLD);

#if defined(USE_MPI_IO)
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
#else
    plist_id = H5P_DEFAULT;
#endif

#else

    if (!file_exists(fname))
        create_hdf5(fname);

    plist_id = H5P_DEFAULT;

#endif

#if defined(USE_MPI) && !defined(USE_MPI_IO)
    for (int itask = 0; itask < mpi_size; ++itask)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (itask != mpi_rank)
            continue;

#endif

        // file_id = H5Fcreate( fname.c_str(), H5F_ACC_RDWR, H5P_DEFAULT,
        // H5P_DEFAULT );
        file_id = H5Fopen(fname.c_str(), H5F_ACC_RDWR, plist_id);

        for (int i = 0; i < 3; ++i)
            count[i] = size(i);

#if defined(USE_MPI)
        count[0] *= mpi_size;
#endif

        if (typeid(data_t) == typeid(float))
            dtype_id = H5T_NATIVE_FLOAT;
        else if (typeid(data_t) == typeid(double))
            dtype_id = H5T_NATIVE_DOUBLE;
        else if (typeid(data_t) == typeid(std::complex<float>))
        {
            dtype_id = H5T_NATIVE_FLOAT;
        }
        else if (typeid(data_t) == typeid(std::complex<double>))
        {
            dtype_id = H5T_NATIVE_DOUBLE;
        }

#if defined(USE_MPI) && !defined(USE_MPI_IO)
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

        hsize_t slice_sz = size(1) * size(2);

        real_t *buf = new real_t[slice_sz];

        count[0] = 1;
        count[1] = size(1);
        count[2] = size(2);

        offset[1] = 0;
        offset[2] = 0;

#if defined(USE_MPI) && defined(USE_MPI_IO)
        H5Pclose(plist_id);
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#else
    plist_id = H5P_DEFAULT;
#endif

        memspace = H5Screate_simple(3, count, NULL);
        filespace = H5Dget_space(dset_id);

        for (size_t i = 0; i < size(0); ++i)
        {
#if defined(USE_MPI)
            offset[0] = mpi_rank * size(0) + i;
#else
        offset[0] = i;
#endif

            for (size_t j = 0; j < size(1); ++j)
            {
                for (size_t k = 0; k < size(2); ++k)
                {
                    if( this->space_ == rspace_id )
                        buf[j * size(2) + k] = std::real(relem(i, j, k));
                    else
                        buf[j * size(2) + k] = std::real(kelem(i, j, k));
                }
            }

            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
            H5Dwrite(dset_id, dtype_id, memspace, filespace, H5P_DEFAULT, buf);
        }

        H5Sclose(filespace);
        H5Sclose(memspace);

#if defined(USE_MPI) && defined(USE_MPI_IO)
        H5Pclose(plist_id);
#endif

        // H5Sclose(filespace);
        H5Dclose(dset_id);

        if (typeid(data_t) == typeid(std::complex<float>) ||
            typeid(data_t) == typeid(std::complex<double>) ||
            this->space_ == kspace_id )
        {
            datasetname += std::string(".im");

            for (int i = 0; i < 3; ++i)
                count[i] = size(i);
#if defined(USE_MPI)
            count[0] *= mpi_size;
#endif

#if defined(USE_MPI) && !defined(USE_MPI_IO)
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

#if defined(USE_MPI) && defined(USE_MPI_IO)
            // H5Pclose( plist_id );
            plist_id = H5Pcreate(H5P_DATASET_XFER);
            H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#else
        plist_id = H5P_DEFAULT;
#endif

            count[0] = 1;

            for (size_t i = 0; i < size(0); ++i)
            {
#if defined(USE_MPI)
                offset[0] = mpi_rank * size(0) + i;
#else
            offset[0] = i;
#endif

                for (size_t j = 0; j < size(1); ++j)
                    for (size_t k = 0; k < size(2); ++k)
                    {
                        if( this->space_ == rspace_id )
                            buf[j * size(2) + k] = std::imag(relem(i, j, k));
                        else
                            buf[j * size(2) + k] = std::imag(kelem(i, j, k));
                    }

                memspace = H5Screate_simple(3, count, NULL);
                filespace = H5Dget_space(dset_id);

                H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count,
                                    NULL);

                H5Dwrite(dset_id, dtype_id, memspace, filespace, H5P_DEFAULT, buf);

                H5Sclose(memspace);
                H5Sclose(filespace);
            }

#if defined(USE_MPI) && defined(USE_MPI_IO)
            H5Pclose(plist_id);
#endif

            H5Dclose(dset_id);

            delete[] buf;
        }

        H5Fclose(file_id);

#if defined(USE_MPI) && !defined(USE_MPI_IO)
    }
#endif
}

#include <iomanip>

template <typename data_t,bool bdistributed>
void Grid_FFT<data_t,bdistributed>::Write_PDF(std::string ofname, int nbins, double scale, double vmin, double vmax)
{
    double logvmin = std::log10(vmin);
    double logvmax = std::log10(vmax);
    double idv = double(nbins) / (logvmax - logvmin);

    std::vector<double> count(nbins, 0.0), scount(nbins, 0.0);

    for (size_t ix = 0; ix < size(0); ix++)
        for (size_t iy = 0; iy < size(1); iy++)
            for (size_t iz = 0; iz < size(2); iz++)
            {
                auto v = this->relem(ix, iy, iz);
                int ibin = int((std::log10(std::abs(v)) - logvmin) * idv);
                if (ibin >= 0 && ibin < nbins)
                {
                    count[ibin] += 1.0;
                }
                ibin = int(((std::log10((std::abs(v) - 1.0) * scale + 1.0)) - logvmin) * idv);
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

        for (int ibin = 0; ibin < nbins; ++ibin)
        {
            double vmean = std::pow(10.0, logvmin + (double(ibin) + 0.5) / idv);
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

template <typename data_t,bool bdistributed>
void Grid_FFT<data_t,bdistributed>::Write_PowerSpectrum(std::string ofname)
{
    std::vector<double> bin_k, bin_P, bin_eP;
    std::vector<size_t> bin_count;
    this->Compute_PowerSpectrum(bin_k, bin_P, bin_eP, bin_count );
#if defined(USE_MPI)
    if (CONFIG::MPI_task_rank == 0)
    {
#endif
        std::ofstream ofs(ofname.c_str());

        ofs << "# " << std::setw(14) << "k" << std::setw(16) << "P(k)" << std::setw(16) << "err. P(k)"
            << std::setw(16) << "#modes"
            << "\n";

        for (size_t ibin = 0; ibin < bin_k.size(); ++ibin)
        {
            if (bin_count[ibin] > 0)
                ofs << std::setw(16) << bin_k[ibin]
                    << std::setw(16) << bin_P[ibin]
                    << std::setw(16) << bin_eP[ibin]
                    << std::setw(16) << bin_count[ibin]
                    << std::endl;
        }
#if defined(USE_MPI)
    }
#endif
}

template <typename data_t,bool bdistributed>
void Grid_FFT<data_t,bdistributed>::Compute_PowerSpectrum(std::vector<double> &bin_k, std::vector<double> &bin_P, std::vector<double> &bin_eP, std::vector<size_t> &bin_count )
{
    this->FourierTransformForward();

    real_t kmax = std::max(std::max(kfac_[0] * nhalf_[0], kfac_[1] * nhalf_[1]),
                           kfac_[2] * nhalf_[2]),
           kmin = std::min(std::min(kfac_[0], kfac_[1]), kfac_[2]),
           dk = kmin;

    const int nbins = kmax / kmin;

    bin_count.assign(nbins, 0);
    bin_k.assign(nbins, 0);
    bin_P.assign(nbins, 0);
    bin_eP.assign(nbins, 0);

    for (size_t ix = 0; ix < size(0); ix++)
        for (size_t iy = 0; iy < size(1); iy++)
            for (size_t iz = 0; iz < size(2); iz++)
            {
                vec3_t<double> k3 = get_k<double>(ix, iy, iz);
                double k = k3.norm();
                int idx2 = k / dk; //int((1.0f / dklog * std::log10(k / kmin)));
                auto z = this->kelem(ix, iy, iz);
                double vabs = z.real() * z.real() + z.imag() * z.imag();

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
    const real_t fftfac(this->fft_norm_fac_ * this->fft_norm_fac_);

    for (int i = 0; i < nbins; ++i)
    {
        if (bin_count[i] > 0)
        {
            bin_k[i] /= bin_count[i];
            bin_P[i] = bin_P[i] / bin_count[i] * volfac * fftfac;
            bin_eP[i] = std::sqrt(bin_eP[i] / bin_count[i] - bin_P[i] * bin_P[i]) / std::sqrt(bin_count[i]) * volfac * fftfac;
        }
    }
}

/********************************************************************************************/

template class Grid_FFT<real_t,true>;
template class Grid_FFT<real_t,false>;
template class Grid_FFT<ccomplex_t,true>;
template class Grid_FFT<ccomplex_t,false>;
