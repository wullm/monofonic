#pragma once

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <numeric>
#include <vector>
#include <general.hh>

namespace particle{
	
class container
{
public:
	std::vector<float> positions_, velocities_;
	std::vector<int> ids_;

	container()
	{
	}

	container(const container &) = delete;

	const void* get_pos_ptr() const{
		return reinterpret_cast<const void*>( &positions_[0] );
	}

	const void* get_vel_ptr() const{
		return reinterpret_cast<const void*>( &velocities_[0] );
	}

	const void* get_ids_ptr() const{
		return reinterpret_cast<const void*>( &ids_[0] );
	}

	void allocate(size_t nump)
	{
		positions_.resize(3 * nump);
		velocities_.resize(3 * nump);
		ids_.resize(nump);
	}

	void set_pos(size_t ipart, size_t idim, real_t p)
	{
		positions_[3 * ipart + idim] = p;
	}

	void set_vel(size_t ipart, size_t idim, real_t p)
	{
		velocities_[3 * ipart + idim] = p;
	}

	void set_id(size_t ipart, id_t id)
	{
		ids_[ipart] = id;
	}

	size_t get_local_num_particles(void) const
	{
		return ids_.size();
	}

	size_t get_global_num_particles(void) const
	{
		size_t local_nump = ids_.size(), global_nump;
#ifdef USE_MPI
		MPI_Allreduce(reinterpret_cast<void *>(&local_nump), reinterpret_cast<void *>(&global_nump), 1,
					  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#else
		global_nump = local_nump;
#endif
		return global_nump;
	}

	size_t get_local_offset( void ) const
	{
		size_t this_offset = 0;

		#ifdef USE_MPI
			int mpi_size, mpi_rank;
			MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
			MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
			
			std::vector<size_t> nump_p_task(mpi_size,0), off_p_task;
			size_t num_p_this_task = this->get_local_num_particles();
			MPI_Allgather( reinterpret_cast<const void*>(&num_p_this_task), 1, MPI_UNSIGNED_LONG_LONG,
				reinterpret_cast<void*>(&nump_p_task[0]), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD );

			off_p_task.push_back( 0 );
			std::partial_sum(nump_p_task.begin(), nump_p_task.end(), std::back_inserter(off_p_task) );
			this_offset = nump_p_task.at(mpi_rank);
		#endif

		return this_offset;
	}

	void dump(void)
	{
		for (size_t i = 0; i < ids_.size(); ++i)
		{
			std::cout << positions_[3 * i + 0] << " " << positions_[3 * i + 1] << " " << positions_[3 * i + 2] << " "
					  << velocities_[3 * i + 0] << " " << velocities_[3 * i + 1] << " " << velocities_[3 * i + 2] << std::endl;
		}
	}
};

}