#pragma once

template< typename T >
class vec3{
private:
    std::array<T,3> data_;

public:    
    T &operator[](size_t i){ return data_[i];}
    
    const T &operator[](size_t i) const { return data_[i]; }
    
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
