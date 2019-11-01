#pragma once

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
    
    //! move constructor
    vec3( vec3<T> &&v)
    : data_(std::move(v.data_)), x(data_[0]), y(data_[1]), z(data_[2]){}

    //! construct from initialiser list
    template<typename ...E>
    vec3(E&&...e) 
    : data_{{std::forward<E>(e)...}}, x(data_[0]), y(data_[1]), z(data_[2]){}
    
    T &operator[](size_t i){ return data_[i];}
    
    const T &operator[](size_t i) const { return data_[i]; }

    vec3<T> operator+( const vec3<T>& v ) const{ return vec3({x+v.x,y+v.y,z+v.z}); }

    vec3<T> operator-( const vec3<T>& v ) const{ return vec3({x-v.x,y-v.y,z-v.z}); }

    vec3<T> operator*( T s ) const{ return vec3({x*s,y*s,z*s}); }

    vec3<T>& operator+=( const vec3<T>& v ) const{ x+=v.x; y+=v.y; z+=v.z; return *this; }

    vec3<T>& operator-=( const vec3<T>& v ) const{ x-=v.x; y-=v.y; z-=v.z; return *this; }

    vec3<T>& operator*=( T s ) const{ x*=s; y*=s; z*=s; return *this; }
    
    T dot(const vec3<T> &a) const 
    {
        return data_[0] * a.data_[0] + data_[1] * a.data_[1] + data_[2] * a.data_[2];
    }
    
    T norm_squared(void) const
    {
        return this->dot(*this);
    }

    T norm(void) const
    {
        return std::sqrt( this->norm_squared() );
    }

    
};
