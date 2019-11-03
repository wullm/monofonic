#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>

#include <vec3.hh>

template<typename T>
class mat3s{
protected:
    std::array<T,9> data_;
    gsl_matrix_view m_;
    gsl_vector *eval_;
    gsl_matrix *evec_;
	gsl_eigen_symmv_workspace * wsp_;
						
    void init_gsl(){
        m_ = gsl_matrix_view_array (&data_[0], 3, 3);
        eval_ = gsl_vector_alloc (3);
        evec_ = gsl_matrix_alloc (3, 3);
        wsp_ = gsl_eigen_symmv_alloc (3);
    }

    void free_gsl(){
        gsl_eigen_symmv_free (wsp_);
        gsl_vector_free (eval_);
        gsl_matrix_free (evec_);
    }

public:

    mat3s(){
        this->init_gsl();
    }

    //! copy constructor
    mat3s( const mat3s<T> &m)
    : data_(m.data_){
        this->init_gsl();
    }
    
    //! move constructor
    mat3s( mat3s<T> &&m)
    : data_(std::move(m.data_)){
        this->init_gsl();
    }

    //! construct vec3 from initializer list
    template<typename ...E>
    mat3s(E&&...e) 
    : data_{{std::forward<E>(e)...}}{
        // resort into symmetrix matrix
        data_[8] = data_[5];
        data_[7] = data_[4];
        data_[6] = data_[2];
        data_[5] = data_[4];
        data_[4] = data_[3];
        data_[3] = data_[1];
        this->init_gsl();
    }

    mat3s<T>& operator=(const mat3s<T>& m){
        data_ = m.data_;
        return *this;
    }

    mat3s<T>& operator=(const mat3s<T>&& m){
        data_ = std::move(m.data_);
        return *this;
    }
    
    //! bracket index access to vector components
    T &operator[](size_t i){ return data_[i];}
    
    //! const bracket index access to vector components
    const T &operator[](size_t i) const { return data_[i]; }

    //! matrix 2d index access
    T &operator()(size_t i, size_t j){ return data_[3*i+j]; }

    //! const matrix 2d index access
    const T &operator()(size_t i, size_t j) const { return data_[3*i+j]; }

    //! destructor
    ~mat3s(){
        this->free_gsl();
    }

    void eigen( vec3<T>& evals, vec3<T>& evec1, vec3<T>& evec2, vec3<T>& evec3 ){
        gsl_eigen_symmv (&m_.matrix, eval_, evec_, wsp_);
        gsl_eigen_symmv_sort (eval_, evec_, GSL_EIGEN_SORT_VAL_ASC);

        for( int i=0; i<3; ++i ){
            evals[i] = gsl_vector_get( eval_, i );
            evec1[i] = gsl_matrix_get( evec_, 0, i );
            evec2[i] = gsl_matrix_get( evec_, 1, i );
            evec3[i] = gsl_matrix_get( evec_, 2, i );
        }
    }
};