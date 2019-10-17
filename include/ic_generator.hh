#pragma once

#include <memory>

#include <config_file.hh>
#include <random_plugin.hh>
#include <output_plugin.hh>
#include <cosmology_calculator.hh>

namespace ic_generator{

    enum particle_lattice{
        lattice_sc, lattice_bcc, lattice_fcc
    };

    int Run( ConfigFile& the_config );
    
    int Initialise( ConfigFile& the_config );

    extern std::unique_ptr<RNG_plugin> the_random_number_generator;
    extern std::unique_ptr<output_plugin> the_output_plugin;
    extern std::unique_ptr<CosmologyCalculator>  the_cosmo_calc;

}
