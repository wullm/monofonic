#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>

#include <vec3.hh>

template<typename T>
class mat3{
protected:
    std::array<T,9> data_;
    gsl_matrix_view m_;
    gsl_vector *eval_;
    gsl_matrix *evec_;
	gsl_eigen_symmv_workspace * wsp_;
    bool bdid_alloc_gsl_;
						
    void init_gsl(){
        // allocate memory for GSL operations if we haven't done so yet
        if( !bdid_alloc_gsl_ )
        {
            m_ = gsl_matrix_view_array (&data_[0], 3, 3);
            eval_ = gsl_vector_alloc (3);
            evec_ = gsl_matrix_alloc (3, 3);
            wsp_ = gsl_eigen_symmv_alloc (3);
            bdid_alloc_gsl_ = true;
        }
    }

    void free_gsl(){
        // free memory for GSL operations if it was allocated
        if( bdid_alloc_gsl_ )
        {
            gsl_eigen_symmv_free (wsp_);
            gsl_vector_free (eval_);
            gsl_matrix_free (evec_);
        }
    }

public:

    mat3()
    : bdid_alloc_gsl_(false) 
    {}

    //! copy constructor
    mat3( const mat3<T> &m)
    : data_(m.data_), bdid_alloc_gsl_(false) 
    {}
    
    //! move constructor
    mat3( mat3<T> &&m)
    : data_(std::move(m.data_)), bdid_alloc_gsl_(false) 
    {}

    //! construct mat3 from initializer list
    template<typename ...E>
    mat3(E&&...e) 
    : data_{{std::forward<E>(e)...}}, bdid_alloc_gsl_(false)
    {}

    mat3<T>& operator=(const mat3<T>& m) noexcept{
        data_ = m.data_;
        return *this;
    }

    mat3<T>& operator=(const mat3<T>&& m) noexcept{
        data_ = std::move(m.data_);
        return *this;
    }

    //! destructor
    ~mat3(){
        this->free_gsl();
    }
    
    //! bracket index access to vector components
    T &operator[](size_t i) noexcept { return data_[i];}
    
    //! const bracket index access to vector components
    const T &operator[](size_t i) const noexcept { return data_[i]; }

    //! matrix 2d index access
    T &operator()(size_t i, size_t j) noexcept { return data_[3*i+j]; }

    //! const matrix 2d index access
    const T &operator()(size_t i, size_t j) const noexcept { return data_[3*i+j]; }

    //! in-place addition
    mat3<T>& operator+=( const mat3<T>& rhs ) noexcept{
        for (size_t i = 0; i < 9; ++i) {
           (*this)[i] += rhs[i];
        }
        return *this;
    }

    //! in-place subtraction
    mat3<T>& operator-=( const mat3<T>& rhs ) noexcept{
        for (size_t i = 0; i < 9; ++i) {
           (*this)[i] -= rhs[i];
        }
        return *this;
    }

    void zero() noexcept{
        for (size_t i = 0; i < 9; ++i) data_[i]=0;
    }

    void eigen( vec3<T>& evals, vec3<T>& evec1, vec3<T>& evec2, vec3<T>& evec3 )
    {
        // for( auto x : data_ ){
        //     std::cerr << x << " " ;
        // }
        // std::cerr << std::endl;
        // resort into symmetrix matrix
        // data_[8] = data_[5];
        // data_[7] = data_[4];
        // data_[6] = data_[2];
        // data_[5] = data_[4];
        // data_[4] = data_[3];
        // data_[3] = data_[1];

        this->init_gsl();

        gsl_eigen_symmv (&m_.matrix, eval_, evec_, wsp_);
        gsl_eigen_symmv_sort (eval_, evec_, GSL_EIGEN_SORT_VAL_ASC);

        for( int i=0; i<3; ++i ){
            evals[i] = gsl_vector_get( eval_, i );
            evec1[i] = gsl_matrix_get( evec_, i, 0 );
            evec2[i] = gsl_matrix_get( evec_, i, 1 );
            evec3[i] = gsl_matrix_get( evec_, i, 2 );
        }

        // std::cerr << "(" << evals[0] << " " << evals[1] << " " << evals[2] << ")" << std::endl;
    }
};

template<typename T>
constexpr const mat3<T> operator+(const mat3<T> &lhs, const mat3<T> &rhs) noexcept
{
    mat3<T> result;
    for (size_t i = 0; i < 9; ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

// matrix - vector multiplication
template<typename T>
vec3<T> operator*( const mat3<T> &A, const vec3<T> &v ) noexcept
{
    vec3<T> result;
    for( int mu=0; mu<3; ++mu ){
        result[mu] = 0.0;
        for( int nu=0; nu<3; ++nu ){
            result[mu] += A(mu,nu)*v[nu];
        }
    }
    return result;
}
