/*******************************************************************\
 vec3.hh - This file is part of MUSIC2 -
 a code to generate initial conditions for cosmological simulations 
 
 CHANGELOG (only majors, for details see repo):
    06/2019 - Oliver Hahn - first implementation
\*******************************************************************/
#pragma once

//! implements a simple class of 3-vectors of arbitrary scalar type
template< typename T >
class vec3{
private:
    //! holds the data
    std::array<T,3> data_;
    
public: 
    //! expose access to elements via references
    T &x,&y,&z;

    //! empty constructor
    vec3()
    : x(data_[0]),y(data_[1]),z(data_[2]){}

    //! copy constructor
    vec3( const vec3<T> &v)
    : data_(v.data_), x(data_[0]),y(data_[1]),z(data_[2]){}

    //! copy constructor for non-const reference, needed to avoid variadic template being called for non-const reference
    vec3( vec3<T>& v)
    : data_(v.data_), x(data_[0]),y(data_[1]),z(data_[2]){}

    //! move constructor
    vec3( vec3<T> &&v)
    : data_(std::move(v.data_)), x(data_[0]), y(data_[1]), z(data_[2]){}

    //! construct vec3 from initializer list
    template<typename ...E>
    vec3(E&&...e) 
    : data_{{std::forward<E>(e)...}}, x{data_[0]}, y{data_[1]}, z{data_[2]}
    {}
    // vec3( T a, T b, T c ) 
    // : data_{{a,b,c}}, x(data_[0]), y(data_[1]), z(data_[2]){}
    
    //! bracket index access to vector components
    T &operator[](size_t i) noexcept{ return data_[i];}
    
    //! const bracket index access to vector components
    const T &operator[](size_t i) const noexcept { return data_[i]; }

    // assignment operator
    vec3<T>& operator=( const vec3<T>& v ) noexcept { data_=v.data_; return *this; }

    //! implementation of summation of vec3
    vec3<T> operator+( const vec3<T>& v ) const noexcept{ return vec3<T>({x+v.x,y+v.y,z+v.z}); }

    //! implementation of difference of vec3
    vec3<T> operator-( const vec3<T>& v ) const noexcept{ return vec3<T>({x-v.x,y-v.y,z-v.z}); }

    //! implementation of unary negative
    vec3<T> operator-() const noexcept{ return vec3<T>({-x,-y,-z}); }

    //! implementation of scalar multiplication
    vec3<T> operator*( T s ) const noexcept{ return vec3<T>({x*s,y*s,z*s}); }

    //! implementation of scalar division
    vec3<T> operator/( T s ) const noexcept{ return vec3<T>({x/s,y/s,z/s}); }

    //! implementation of += operator
    vec3<T>& operator+=( const vec3<T>& v ) noexcept{ x+=v.x; y+=v.y; z+=v.z; return *this; }

    //! implementation of -= operator
    vec3<T>& operator-=( const vec3<T>& v ) noexcept{ x-=v.x; y-=v.y; z-=v.z; return *this; }

    //! multiply with scalar
    vec3<T>& operator*=( T s ) noexcept{ x*=s; y*=s; z*=s; return *this; }
    
    //! divide by scalar
    vec3<T>& operator/=( T s ) noexcept{ x/=s; y/=s; z/=s; return *this; }

    //! compute dot product with another vector
    T dot(const vec3<T> &a) const noexcept
    {
        return data_[0] * a.data_[0] + data_[1] * a.data_[1] + data_[2] * a.data_[2];
    }
    
    //! returns 2-norm squared of vector
    T norm_squared(void) const noexcept { return this->dot(*this); }

    //! returns 2-norm of vector
    T norm(void) const noexcept { return std::sqrt( this->norm_squared() ); }

    //! wrap absolute vector to box of size p
    vec3<T>& wrap_abs( T p = 1.0 ) noexcept{
        for( auto& x : data_ ) x = std::fmod( 2*p + x, p );
        return *this;
    }

    //! wrap relative vector to box of size p
    vec3<T>& wrap_rel( T p = 1.0 ) noexcept{
        for( auto& x : data_ ) x = (x<-p/2)? x+p : (x>=p/2)? x-p : x;
        return *this;
    }
};

//! multiplication with scalar
template<typename T>
vec3<T> operator*( T s, const vec3<T>& v ){
    return vec3<T>({v.x*s,v.y*s,v.z*s});
}
