#pragma once
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"

namespace dyno{

	#define INVALID -1
	#define BUCKETS 8
	#define CAPACITY 16

	template<typename TDataType>
	class GridHash
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		GridHash();
		~GridHash();

		void setSpace(Real _h, Coord _lo, Coord _hi);

		void construct(DArray<Coord>& pos);

		void clear();

		void release();

		GPU_FUNC inline int getIndex(int i, int j, int k)
		{
			if (i < 0 || i >= nx) return INVALID;
			if (j < 0 || j >= ny) return INVALID;
			if (k < 0 || k >= nz) return INVALID;

			return i + j*nx + k*nx*ny;
		}

		GPU_FUNC inline int getIndex(Coord pos) {
			int i = (int)floor((pos[0] - lo[0]) / ds);
			int j = (int)floor((pos[1] - lo[1]) / ds);
			int k = (int)floor((pos[2] - lo[2]) / ds);

			return getIndex(i, j, k);
		}

		GPU_FUNC inline int3 getIndex3(Coord pos) {
			int i = (int)floor((pos[0] - lo[0]) / ds);
			int j = (int)floor((pos[1] - lo[1]) / ds);
			int k = (int)floor((pos[2] - lo[2]) / ds);

			return make_int3(i, j, k);
		}

		GPU_FUNC inline int getCounter(int gId) { 
			if (gId >= num - 1) {
				return particle_num - index[gId];
			}

			return index[gId + 1] - index[gId];
		}

		GPU_FUNC inline int getParticleId(int gId, int n) { 
			return ids[index[gId] + n];
		}

	public:
		int num;
		int nx, ny, nz;

		int particle_num = 0;

		Real ds;

		Coord lo;
		Coord hi;

		//int npMax;		//maximum particle number for each cell

		int* ids = nullptr;
		int* counter = nullptr;
		int* index = nullptr;

		Scan<int>* m_scan = nullptr;
		Reduction<int>* m_reduce = nullptr;
	};
}
