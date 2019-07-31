/*
 
 output_generic.cc - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010-13  Oliver Hahn
 
 */


#ifdef USE_HDF5

#include "output_plugin.hh"
#include "HDF_IO.hh"


class generic_output_plugin : public output_plugin
{
protected:
	
	using output_plugin::cf_;

	const unsigned levelmax_{0};
		
	// template< typename Tt >
	//void write2HDF5( std::string fname, std::string dname, const MeshvarBnd<Tt>& data )
	void write2HDF5( std::string fname, std::string dname, const grid_hierarchy& gh, unsigned ilevel )
	{
		unsigned n0 = gh.get_grid(ilevel)->size(0);
		unsigned n1 = gh.get_grid(ilevel)->size(1);
		unsigned n2 = gh.get_grid(ilevel)->size(2);

		std::vector<double> temp_data;
		temp_data.reserve(size_t(n0)*size_t(n1)*size_t(n2));
		
		for (unsigned i = 0; i < n0; ++i){
			for (unsigned j = 0; j < n1; ++j){
				for (unsigned k = 0; k < n2; ++k){
					temp_data.push_back((*gh.get_grid(ilevel)).relem(i, j, k));
				}
			}
		}

		unsigned nd[3] = {n0,n1,n2};//{ (unsigned)(n0+2*nb),(unsigned)(n1+2*nb),(unsigned)(n2+2*nb)	};
		HDFWriteDataset3D( fname, dname, nd, temp_data);
	}
	
public:
	generic_output_plugin( ConfigFile& cf )//std::string fname, Cosmology cosm, Parameters param )
	: output_plugin( cf )//fname, cosm, param )
	{

		HDFCreateFile(fname_);
		
		HDFCreateGroup(fname_, "header");

		HDFWriteDataset(fname_,"/header/grid_off_x",offx_);
		HDFWriteDataset(fname_,"/header/grid_off_y",offy_);
		HDFWriteDataset(fname_,"/header/grid_off_z",offz_);
		
		HDFWriteDataset(fname_,"/header/grid_len_x",sizex_);
		HDFWriteDataset(fname_,"/header/grid_len_y",sizey_);
		HDFWriteDataset(fname_,"/header/grid_len_z",sizez_);
		
		// HDFWriteGroupAttribute(fname_, "header", "levelmin", levelmin_ );
		// HDFWriteGroupAttribute(fname_, "header", "levelmax", levelmax_ );
	}
	
	~generic_output_plugin()
	{	}
	
	void write_dm_mass( const grid_hierarchy& gh )
	{	}
	
	void write_dm_velocity( int coord, const grid_hierarchy& gh )
	{
		char sstr[128];
		
		for( unsigned ilevel=0; ilevel<=levelmax_; ++ilevel )
		{
			if( coord == 0 )
				sprintf(sstr,"level_%03d_DM_vx",ilevel);
			else if( coord == 1 )
				sprintf(sstr,"level_%03d_DM_vy",ilevel);
			else if( coord == 2 )
				sprintf(sstr,"level_%03d_DM_vz",ilevel);
			
			write2HDF5( fname_, sstr, gh, ilevel );
		}
	}
	
	void write_dm_position( int coord, const grid_hierarchy& gh )
	{
		char sstr[128];
		
		for( unsigned ilevel=0; ilevel<=levelmax_; ++ilevel )
		{
			if( coord == 0 )
				sprintf(sstr,"level_%03d_DM_dx",ilevel);
			else if( coord == 1 )
				sprintf(sstr,"level_%03d_DM_dy",ilevel);
			else if( coord == 2 )
				sprintf(sstr,"level_%03d_DM_dz",ilevel);
			
			write2HDF5( fname_, sstr, gh, ilevel );
		}
	}
	
	void write_dm_density( const grid_hierarchy& gh )
	{
		char sstr[128];
		
		for( unsigned ilevel=0; ilevel<=levelmax_; ++ilevel )
		{
			sprintf(sstr,"level_%03d_DM_rho",ilevel);
			write2HDF5( fname_, sstr, gh, ilevel );
		}

		// double h = 1.0/(1<<levelmin_);
		// double shift[3];
		// shift[0] = -(double)cf_.GetValue<int>( "setup", "shift_x" )*h;
		// shift[1] = -(double)cf_.GetValue<int>( "setup", "shift_y" )*h;
		// shift[2] = -(double)cf_.GetValue<int>( "setup", "shift_z" )*h;
			
	}
	
	void write_dm_potential( const grid_hierarchy& gh )
	{ 
		char sstr[128];
		
		for( unsigned ilevel=0; ilevel<=levelmax_; ++ilevel )
		{
			sprintf(sstr,"level_%03d_DM_potential",ilevel);
			write2HDF5( fname_, sstr, gh, ilevel );
		}
	}
	
	void write_gas_potential( const grid_hierarchy& gh )
	{ 
		char sstr[128];
		
		for( unsigned ilevel=0; ilevel<=levelmax_; ++ilevel )
		{
			sprintf(sstr,"level_%03d_BA_potential",ilevel);
			write2HDF5( fname_, sstr, gh, ilevel );
		}
	}
	
	
	
	void write_gas_velocity( int coord, const grid_hierarchy& gh )
	{	
		char sstr[128];
		
		for( unsigned ilevel=0; ilevel<=levelmax_; ++ilevel )
		{
			if( coord == 0 )
				sprintf(sstr,"level_%03d_BA_vx",ilevel);
			else if( coord == 1 )
				sprintf(sstr,"level_%03d_BA_vy",ilevel);
			else if( coord == 2 )
				sprintf(sstr,"level_%03d_BA_vz",ilevel);
			
			write2HDF5( fname_, sstr, gh, ilevel );
		}
	}
	
	void write_gas_position( int coord, const grid_hierarchy& gh )
	{	}
	
	void write_gas_density( const grid_hierarchy& gh )
	{	
		char sstr[128];
		
		for( unsigned ilevel=0; ilevel<=levelmax_; ++ilevel )
		{
			sprintf(sstr,"level_%03d_BA_rho",ilevel);
			write2HDF5( fname_, sstr, gh, ilevel );
		}
	}
	
	void finalize( void )
	{	}
};



namespace{
	output_plugin_creator_concrete< generic_output_plugin > creator("generic");
}

#endif

