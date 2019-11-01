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
    csoca::ilog << "Available random number generator plug-ins:" << std::endl;
    while (it != m.end())
    {
        if ((*it).second){
            csoca::ilog.Print("\t\'%s\'\n", (*it).first.c_str());
        }
        ++it;
    }
    csoca::ilog << std::endl;
}

std::unique_ptr<RNG_plugin> select_RNG_plugin(ConfigFile &cf)
{
    std::string rngname = cf.GetValueSafe<std::string>("random", "generator", "MUSIC");

    RNG_plugin_creator *the_RNG_plugin_creator = get_RNG_plugin_map()[rngname];

    if (!the_RNG_plugin_creator)
    {
        csoca::ilog.Print("Invalid/Unregistered random number generator plug-in encountered : %s", rngname.c_str());
        print_RNG_plugins();
        throw std::runtime_error("Unknown random number generator plug-in");
    }
    else
    {
        csoca::ilog << "-------------------------------------------------------------------------------" << std::endl;
        csoca::ilog << std::setw(32) << std::left << "Random number generator plugin" << " : " << rngname << std::endl;
    }

    return std::move(the_RNG_plugin_creator->Create(cf));
}