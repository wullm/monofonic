#pragma once

#include <map>
#include <config_file.hh>
#include <general.hh>
#include <grid_fft.hh>

#define DEF_RAN_CUBE_SIZE 32

class RNG_plugin
{
  protected:
    config_file *pcf_; //!< pointer to config_file from which to read parameters
  public:
    explicit RNG_plugin(config_file &cf)
        : pcf_(&cf)
    {
    }
    virtual ~RNG_plugin() {}
    virtual bool isMultiscale() const = 0;
    virtual void Fill_Grid( Grid_FFT<real_t>& g ) = 0;//const = 0;
    //virtual void FillGrid(int level, DensityGrid<real_t> &R) = 0;
};

struct RNG_plugin_creator
{
    virtual std::unique_ptr<RNG_plugin> Create(config_file &cf) const = 0;
    virtual ~RNG_plugin_creator() {}
};

std::map<std::string, RNG_plugin_creator *> & get_RNG_plugin_map();

void print_RNG_plugins(void);

template <class Derived>
struct RNG_plugin_creator_concrete : public RNG_plugin_creator
{
    //! register the plugin by its name
    RNG_plugin_creator_concrete(const std::string &plugin_name)
    {
        get_RNG_plugin_map()[plugin_name] = this;
    }

    //! create an instance of the plugin
    std::unique_ptr<RNG_plugin> Create(config_file &cf) const
    {
        return std::make_unique<Derived>(cf);
    }
};

typedef RNG_plugin RNG_instance;
std::unique_ptr<RNG_plugin> select_RNG_plugin( config_file &cf);

// /*!
//  * @brief encapsulates all things for multi-scale white noise generation
//  */
// template <typename T>
// class random_number_generator
// {
//   protected:
//     config_file *pcf_;
//     //const refinement_hierarchy * prefh_;
//     RNG_plugin *generator_;
//     int levelmin_, levelmax_;

//   public:
//     //! constructor
//     random_number_generator( config_file &cf )
//         : pcf_(&cf) //, prefh_( &refh )
//     {
//         levelmin_ = pcf_->get_value<int>("setup", "levelmin");
//         levelmax_ = pcf_->get_value<int>("setup", "levelmax");
//         generator_ = select_RNG_plugin(cf);
//     }

//     //! destructor
//     ~random_number_generator()
//     {
//     }

//     //! initialize_for_grid_structure
//     /*void initialize_for_grid_structure(const refinement_hierarchy &refh)
//     {
//         generator_->initialize_for_grid_structure(refh);
//     }*/

//     //! load random numbers to a new array
//     template <typename array>
//     void load(array &A, int ilevel)
//     {
//         generator_->FillGrid(ilevel, A);
//     }
// };

// typedef random_number_generator<real_t> noise_generator;