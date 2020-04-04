#include <general.hh>
#include <random_plugin.hh>

std::map<std::string, RNG_plugin_creator *> &
get_RNG_plugin_map()
{
    static std::map<std::string, RNG_plugin_creator *> RNG_plugin_map;
    return RNG_plugin_map;
}

void print_RNG_plugins()
{
    std::map<std::string, RNG_plugin_creator *> &m = get_RNG_plugin_map();
    std::map<std::string, RNG_plugin_creator *>::iterator it;
    it = m.begin();
    music::ilog << "Available random number generator plug-ins:" << std::endl;
    while (it != m.end())
    {
        if ((*it).second){
            music::ilog.Print("\t\'%s\'\n", (*it).first.c_str());
        }
        ++it;
    }
    music::ilog << std::endl;
}

std::unique_ptr<RNG_plugin> select_RNG_plugin(config_file &cf)
{
    std::string rngname = cf.get_value_safe<std::string>("random", "generator", "MUSIC");

    RNG_plugin_creator *the_RNG_plugin_creator = get_RNG_plugin_map()[rngname];

    if (!the_RNG_plugin_creator)
    {
        music::ilog.Print("Invalid/Unregistered random number generator plug-in encountered : %s", rngname.c_str());
        print_RNG_plugins();
        throw std::runtime_error("Unknown random number generator plug-in");
    }
    else
    {
        music::ilog << "-------------------------------------------------------------------------------" << std::endl;
        music::ilog << std::setw(32) << std::left << "Random number generator plugin" << " : " << rngname << std::endl;
    }

    return std::move(the_RNG_plugin_creator->Create(cf));
}