#pragma once

#include <array>

#include <general.hh>
#include <grid_fft.hh>

//! convolution class, respecting Orszag's 3/2 rule
template< typename data_t >
class OrszagConvolver
{
protected:
    Grid_FFT<data_t> *f1p_, *f2p_;
#ifdef USE_MPI
    Grid_FFT<data_t> *fMPIbuf_;
#endif
    std::array<size_t,3> np_;
    std::array<real_t,3> length_;
    
    ccomplex_t *crecvbuf_;
    real_t *recvbuf_;
    size_t maxslicesz_;
    std::vector<ptrdiff_t> offsets_, offsetsp_;
    std::vector<size_t> sizes_, sizesp_;

    // ptrdiff_t *offsets_;
	// ptrdiff_t *offsetsp_;
	// ptrdiff_t *sizes_;
	// ptrdiff_t *sizesp_;

private:
    // int get_task( ptrdiff_t index, const ptrdiff_t *offsets, const ptrdiff_t *sizes, const int ntasks ) const
    // {
    //     int itask = 0;
    //     while( itask < ntasks-1 && offsets[itask+1] <= index ) ++itask;
    //     return itask;
    // }

    // get task based on offsets

    int get_task(ptrdiff_t index, const std::vector<ptrdiff_t>& offsets, const std::vector<size_t>& sizes, const int ntasks )
    {
        int itask = 0;
        while (itask < ntasks - 1 && offsets[itask + 1] <= index) ++itask;
        return itask;
    }
	// void pad_insert( const Grid_FFT<data_t> & f, Grid_FFT<data_t> & fp );
	// void unpad( const Grid_FFT<data_t> & fp, Grid_FFT< data_t > & f );

public:

    
    OrszagConvolver( const std::array<size_t, 3> &N, const std::array<real_t, 3> &L )
    : np_({3*N[0]/2,3*N[1]/2,3*N[2]/2}), length_(L)
    {
        //... create temporaries
        f1p_ = new Grid_FFT<data_t>(np_, length_, kspace_id);
        f2p_ = new Grid_FFT<data_t>(np_, length_, kspace_id);

#if defined(USE_MPI)
        fMPIbuf_ = new Grid_FFT<data_t>(N, length_, kspace_id);
        maxslicesz_ = f1p_->sizes_[1] * f1p_->sizes_[3] * 2;

        crecvbuf_ = new ccomplex_t[maxslicesz_ / 2];
        recvbuf_ = reinterpret_cast<real_t *>(&crecvbuf_[0]);

        int ntasks(MPI_Get_size());

        offsets_.assign(ntasks,0);
        offsetsp_.assign(ntasks,0);
        sizes_.assign(ntasks,0);
        sizesp_.assign(ntasks,0);
        // offsets_ = new ptrdiff_t[ntasks];
        // offsetsp_ = new ptrdiff_t[ntasks];
        // sizes_ = new ptrdiff_t[ntasks];
        // sizesp_ = new ptrdiff_t[ntasks];

        size_t tsize = N[0], tsizep = f1p_->size(0);

        MPI_Allgather(&fMPIbuf_->local_1_start_, 1, MPI_LONG_LONG, &offsets_[0], 1,
                        MPI_LONG_LONG, MPI_COMM_WORLD);
        MPI_Allgather(&f1p_->local_1_start_, 1, MPI_LONG_LONG, &offsetsp_[0], 1,
                        MPI_LONG_LONG, MPI_COMM_WORLD);
        MPI_Allgather(&tsize, 1, MPI_LONG_LONG, &sizes_[0], 1, MPI_LONG_LONG,
                        MPI_COMM_WORLD);
        MPI_Allgather(&tsizep, 1, MPI_LONG_LONG, &sizesp_[0], 1, MPI_LONG_LONG,
                        MPI_COMM_WORLD);
#endif
    }

    ~OrszagConvolver()
    {
        delete f1p_;
        delete f2p_;
#if defined(USE_MPI)
        delete fMPIbuf_;
        delete[] crecvbuf_;
        // delete[] offsets_;
        // delete[] offsetsp_;
        // delete[] sizes_;
        // delete[] sizesp_;
#endif
    }

    template< typename opp >
    void convolve_Hessians( Grid_FFT<data_t> & inl, const std::array<int,2>& d2l, Grid_FFT<data_t> & inr, const std::array<int,2>& d2r, Grid_FFT<data_t> & res, opp op ){
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // perform convolution of Hessians
        this->convolve2(
            [&]( size_t i, size_t j, size_t k ) -> ccomplex_t{
                auto kk = inl.template get_k<real_t>(i,j,k);
                return -kk[d2l[0]] * kk[d2l[1]] * inl.kelem(i,j,k);
            },
            [&]( size_t i, size_t j, size_t k ){
                auto kk = inr.template get_k<real_t>(i,j,k);
                return -kk[d2r[0]] * kk[d2r[1]] * inr.kelem(i,j,k);
            }, res, op );
    }

    template< typename opp >
    void convolve_SumHessians( Grid_FFT<data_t> & inl, const std::array<int,2>& d2l, Grid_FFT<data_t> & inr, const std::array<int,2>& d2r1, 
                               const std::array<int,2>& d2r2, Grid_FFT<data_t> & res, opp op ){
        // transform to FS in case fields are not
        inl.FourierTransformForward();
        inr.FourierTransformForward();
        // perform convolution of Hessians
        this->convolve2(
            [&]( size_t i, size_t j, size_t k ) -> ccomplex_t{
                auto kk = inl.template get_k<real_t>(i,j,k);
                return -kk[d2l[0]] * kk[d2l[1]] * inl.kelem(i,j,k);
            },
            [&]( size_t i, size_t j, size_t k ){
                auto kk = inr.template get_k<real_t>(i,j,k);
                return (-kk[d2r1[0]] * kk[d2r1[1]] -kk[d2r2[0]] * kk[d2r2[1]]) * inr.kelem(i,j,k);
            }, res, op );
    }

    template< typename kfunc1, typename kfunc2, typename opp >
    void convolve2( kfunc1 kf1, kfunc2 kf2, Grid_FFT<data_t> & res, opp op )
    {
        //... prepare data 1
        f1p_->FourierTransformForward(false);
        this->pad_insert( kf1, *f1p_ );

        //... prepare data 1
        f2p_->FourierTransformForward(false);
        this->pad_insert( kf2, *f2p_ );

        //... convolve
        f1p_->FourierTransformBackward();
        f2p_->FourierTransformBackward();

        #pragma omp parallel for
        for (size_t i = 0; i < f1p_->ntot_; ++i){
            (*f2p_).relem(i) *= (*f1p_).relem(i);
        }
        f2p_->FourierTransformForward();
        //... copy data back
        res.FourierTransformForward();
        unpad(*f2p_, res, op);
    }

    //... inplace interface
	/*void convolve3( const Grid_FFT<data_t> & f1, const Grid_FFT<data_t> & f2, const Grid_FFT<data_t> & f3, Grid_FFT<data_t> & res )
    {
        convolve2( f1, f2, res );
        convolve2( res, f3, res );
    }*/


private:
    template <typename kdep_functor>
    void pad_insert( kdep_functor kfunc, Grid_FFT<data_t> &fp ){
        assert( fp.space_ == kspace_id );
        

        // size_t dn[3] = {
        //     fp.n_[0]/3,// fp.n_[0] - f.n_[0],
        //     fp.n_[1]/3,// fp.n_[1] - f.n_[1],
        //     fp.n_[2]/3// fp.n_[2] - f.n_[2],
        // };
        const double rfac = std::pow(1.5,1.5);//std::sqrt(fp.n_[0] * fp.n_[1] * fp.n_[2]) / std::sqrt(f.n_[0] * f.n_[1] * f.n_[2]);

        fp.zero();

        
    #if !defined(USE_MPI) ////////////////////////////////////////////////////////////////////////////////////
        //size_t nhalf[3] = {f.n_[0] / 2, f.n_[1] / 2, f.n_[2] / 2};
        size_t nhalf[3] = {fp.n_[0] / 3, fp.n_[1] / 3, fp.n_[2] / 3};

        #pragma omp parallel for
        for (size_t i = 0; i < 2*fp.size(0)/3; ++i)
        {
            size_t ip = (i > nhalf[0]) ? i + nhalf[0] : i;
            for (size_t j = 0; j < 2*fp.size(1)/3; ++j)
            {
                size_t jp = (j > nhalf[1]) ? j + nhalf[1] : j;
                for (size_t k = 0; k < 2*fp.size(2)/3; ++k)
                {
                    size_t kp = (k > nhalf[2]) ? k + nhalf[2] : k;
                    // if( i==nhalf[0]||j==nhalf[1]||k==nhalf[2]) continue;
                    //fp.kelem(ip, jp, kp) = f.kelem(i, j, k) * rfac;
                    fp.kelem(ip, jp, kp) = kfunc(i, j, k) * rfac;
                }
            }
        }

    #else /// then USE_MPI is defined ////////////////////////////////////////////////////////////

        MPI_Barrier(MPI_COMM_WORLD);

        /////////////////////////////////////////////////////////////////////

        double tstart = get_wtime();
        csoca::dlog << "[MPI] Started scatter for convolution" << std::endl;

        //... collect offsets

        assert(fMPIbuf_->space_ == kspace_id);

        size_t nf[3] = {fMPIbuf_->size(0), fMPIbuf_->size(1), fMPIbuf_->size(2)};
        size_t nfp[3] = {fp.size(0), fp.size(1), fp.size(2)};
        size_t fny[3] = {fMPIbuf_->n_[1] / 2, fMPIbuf_->n_[0] / 2, fMPIbuf_->n_[2] / 2};

        //... local size must be divisible by 2, otherwise this gets too complicated
        assert(fMPIbuf_->n_[1] % 2 == 0);

        size_t slicesz = fMPIbuf_->size(1) * fMPIbuf_->size(3); //*2;

        // comunicate
        // check if this is a real field (then we get the wrong size)
        // if (typeid(data_t) == typeid(real_t))
            // slicesz *= 2; // then sizeof(real_t) gives only half of a complex

        // MPI_Datatype datatype =
        //     (typeid(data_t) == typeid(float))
        //         ? MPI_FLOAT
        //         : (typeid(data_t) == typeid(double))
        //             ? MPI_DOUBLE
        //             : (typeid(data_t) == typeid(std::complex<float>))
        //                     ? MPI_COMPLEX
        //                     : (typeid(data_t) == typeid(std::complex<double>))
        //                         ? MPI_DOUBLE_COMPLEX
        //                         : MPI_INT;

        MPI_Datatype datatype = 
            (typeid(data_t) == typeid(float)) ? MPI_COMPLEX : 
            (typeid(data_t) == typeid(double)) ? MPI_DOUBLE_COMPLEX : MPI_BYTE;


        // fill MPI send buffer
        fMPIbuf_->FourierTransformForward(false);
        #pragma omp parallel for
        for (size_t i = 0; i < fMPIbuf_->size(0); ++i)
        {
            for (size_t j = 0; j < fMPIbuf_->size(1); ++j)
            {
                for (size_t k = 0; k < fMPIbuf_->size(2); ++k)
                {
                    fMPIbuf_->kelem(i, j, k) = kfunc(i, j, k) * rfac;
                }
            }
        }

        MPI_Status status;

        std::vector<MPI_Request> req;
        MPI_Request temp_req;

        // send data from buffer
        for (size_t i = 0; i < nf[0]; ++i)
        {
            size_t iglobal = i + offsets_[CONFIG::MPI_task_rank];

            if (iglobal < fny[0])
            {
                int sendto = get_task(iglobal, offsetsp_, sizesp_, CONFIG::MPI_task_size);
                MPI_Isend(&fMPIbuf_->kelem(i * slicesz), (int)slicesz, datatype, sendto,
                        (int)iglobal, MPI_COMM_WORLD, &temp_req);
                req.push_back(temp_req);
                // std::cout << "task " << CONFIG::MPI_task_rank << " : added request No" << req.size()-1 << ": Isend #" << iglobal << " to task " << sendto << ", size = " << slicesz << std::endl;
            }
            if (iglobal > fny[0])
            {
                int sendto = get_task(iglobal + fny[0], offsetsp_, sizesp_, CONFIG::MPI_task_size);
                MPI_Isend(&fMPIbuf_->kelem(i * slicesz), (int)slicesz, datatype, sendto,
                        (int)(iglobal + fny[0]), MPI_COMM_WORLD, &temp_req);
                req.push_back(temp_req);
                // std::cout << "task " << CONFIG::MPI_task_rank << " : added request No" << req.size()-1 << ": Isend #" << iglobal+fny[0] << " to task " << sendto << ", size = " << slicesz<< std::endl;
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

                // std::cout << "task " << CONFIG::MPI_task_rank << " : receive #" << iglobal << " from task "
                // << recvfrom << ", size = " << slicesz << ", " << crecvbuf_ << ", " << datatype << std::endl;

                MPI_Recv(&recvbuf_[0], (int)slicesz, datatype, recvfrom, (int)iglobal,
                        MPI_COMM_WORLD, &status);
                // std::cout << "---> ok!  " << (bool)(status.MPI_ERROR==MPI_SUCCESS) << std::endl;

                // assert(status.MPI_ERROR == MPI_SUCCESS);

                for (size_t j = 0; j < nf[1]; ++j)
                {
                    if (j < fny[1])
                    {
                        size_t jp = j;
                        for (size_t k = 0; k < nf[2]; ++k)
                        {
                            // size_t kp = (k>fny[2])? k+fny[2] : k;
                            if (k < fny[2])
                                fp.kelem(i, jp, k) = crecvbuf_[j * fMPIbuf_->sizes_[3] + k];
                            else if (k > fny[2])
                                fp.kelem(i, jp, k + fny[2]) = crecvbuf_[j * fMPIbuf_->sizes_[3] + k];
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
                                fp.kelem(i, jp, k) = crecvbuf_[j * fMPIbuf_->sizes_[3] + k];
                            else if (k > fny[2])
                                fp.kelem(i, jp, k + fny[2]) = crecvbuf_[j * fMPIbuf_->sizes_[3] + k];
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


    template <typename operator_t>
    void unpad(const Grid_FFT<data_t> &fp, Grid_FFT<data_t> &f, operator_t op )
    {
        // assert(fp.n_[0] == 3 * f.n_[0] / 2);
        // assert(fp.n_[1] == 3 * f.n_[1] / 2);
        // assert(fp.n_[2] == 3 * f.n_[2] / 2);

        // make sure we're in Fourier space...
        assert( fp.space_ == kspace_id );
        f.FourierTransformForward();

    #if !defined(USE_MPI) ////////////////////////////////////////////////////////////////////////////////////
        size_t dn[3] = {
            fp.n_[0] - f.n_[0],
            fp.n_[1] - f.n_[1],
            fp.n_[2] - f.n_[2],
        };
        size_t nhalf[3] = {f.n_[0] / 2, f.n_[1] / 2, f.n_[2] / 2};
        const double rfac = std::sqrt(fp.n_[0] * fp.n_[1] * fp.n_[2]) / std::sqrt(f.n_[0] * f.n_[1] * f.n_[2]);

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
                    f.kelem(i, j, k) = op(fp.kelem(ip, jp, kp) / rfac, f.kelem(i, j, k));
                }
            }
        }

    #else /// then USE_MPI is defined //////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////

        double tstart = get_wtime();

        csoca::ilog << "[MPI] Started gather for convolution";

        MPI_Barrier(MPI_COMM_WORLD);

        size_t nf[3] = {f.size(0), f.size(1), f.size(2)};
        size_t nfp[4] = {fp.size(0), fp.size(1), fp.size(2), fp.size(3)};
        size_t fny[3] = {f.n_[1] / 2, f.n_[0] / 2, f.n_[2] / 2};

        size_t slicesz = fp.size(1) * fp.size(3);

        // if (typeid(data_t) == typeid(real_t))
        //     slicesz *= 2; // then sizeof(real_t) gives only half of a complex

        // MPI_Datatype datatype =
        //     (typeid(data_t) == typeid(float))
        //         ? MPI_FLOAT
        //         : (typeid(data_t) == typeid(double))
        //             ? MPI_DOUBLE
        //             : (typeid(data_t) == typeid(std::complex<float>))
        //                     ? MPI_COMPLEX
        //                     : (typeid(data_t) == typeid(std::complex<double>))
        //                         ? MPI_DOUBLE_COMPLEX
        //                         : MPI_INT;

        MPI_Datatype datatype = 
            (typeid(data_t) == typeid(float)) ? MPI_COMPLEX : 
            (typeid(data_t) == typeid(double)) ? MPI_DOUBLE_COMPLEX : MPI_BYTE;

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
                            f.kelem(i, j, k) = op(crecvbuf_[jp * nfp[3] + k],f.kelem(i, j, k));
                        else if (k > fny[2])
                            f.kelem(i, j, k) = op(crecvbuf_[jp * nfp[3] + k + fny[2]], f.kelem(i, j, k));
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
                            f.kelem(i, j, k) = op(crecvbuf_[jp * nfp[3] + k], f.kelem(i, j, k));
                        else if (k > fny[2])
                            f.kelem(i, j, k) = op(crecvbuf_[jp * nfp[3] + k + fny[2]], f.kelem(i, j, k));
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


};
