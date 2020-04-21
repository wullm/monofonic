
#include <transfer_function_plugin.hh>

std::map<std::string, TransferFunction_plugin_creator *> &
get_TransferFunction_plugin_map()
{
    static std::map<std::string, TransferFunction_plugin_creator *> TransferFunction_plugin_map;
    return TransferFunction_plugin_map;
}

void print_TransferFunction_plugins()
{
    std::map<std::string, TransferFunction_plugin_creator *> &m = get_TransferFunction_plugin_map();
    std::map<std::string, TransferFunction_plugin_creator *>::iterator it;
    it = m.begin();
    music::ilog << "Available transfer function plug-ins:" << std::endl;
    while (it != m.end())
    {
        if ((*it).second)
            music::ilog << "\t\'" << (*it).first << "\'" << std::endl;
        ++it;
    }
    music::ilog << std::endl;
}

std::unique_ptr<TransferFunction_plugin> select_TransferFunction_plugin(config_file &cf)
{
    std::string tfname = cf.get_value<std::string>("cosmology", "transfer");

    TransferFunction_plugin_creator *the_TransferFunction_plugin_creator = get_TransferFunction_plugin_map()[tfname];

    if (!the_TransferFunction_plugin_creator)
    {
        music::elog << "Invalid/Unregistered transfer function plug-in encountered : " << tfname << std::endl;
        print_TransferFunction_plugins();
        throw std::runtime_error("Unknown transfer function plug-in");
    }
    else
    {
        music::ilog << "-------------------------------------------------------------------------------" << std::endl;
        music::ilog << std::setw(32) << std::left << "Transfer function plugin" << " : " << tfname << std::endl;
    }

    return std::move(the_TransferFunction_plugin_creator->create(cf));
}
