#pragma once

#include <memory>

#include <config_file.hh>
#include <random_plugin.hh>
#include <output_plugin.hh>
#include <cosmology_calculator.hh>

namespace ic_generator{

    int Run( config_file& the_config );
    
    int Initialise( config_file& the_config );

    extern std::unique_ptr<RNG_plugin> the_random_number_generator;
    extern std::unique_ptr<output_plugin> the_output_plugin;
    extern std::unique_ptr<cosmology::calculator>  the_cosmo_calc;

}
