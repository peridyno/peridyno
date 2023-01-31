#include "GridHash.h"

#include "Object.h"
#include "DataTypes.h"

namespace dyno 
{
	__constant__ int offset[27][3] = { 0, 0, 0,
		0, 0, 1,
		0, 1, 0,
		1, 0, 0,
		0, 0, -1,
		0, -1, 0,
		-1, 0, 0,
		0, 1, 1,
		0, 1, -1,
		0, -1, 1,
		0, -1, -1,
		1, 0, 1,
		1, 0, -1,
		-1, 0, 1,
		-1, 0, -1,
		1, 1, 0,
		1, -1, 0,
		-1, 1, 0,
		-1, -1, 0,
		1, 1, 1,
		1, 1, -1,
		1, -1, 1,
		-1, 1, 1,
		1, -1, -1,
		-1, 1, -1,
		-1, -1, 1,
		-1, -1, -1
	};

	template<typename TDataType>
	GridHash<TDataType>::GridHash()
	{
	}

	template<typename TDataType>
	GridHash<TDataType>::~GridHash()
	{
	}

	template<typename TDataType>
	void GridHash<TDataType>::setSpace(Real _h, Coord _lo, Coord _hi)
	{
		release();

		int padding = 2;
		ds = _h;
		lo = _lo - padding * ds;

		Coord nSeg = (_hi - _lo) / ds;

		nx = (int)ceil(nSeg[0]) + 1 + 2 * padding;
		ny = (int)ceil(nSeg[1]) + 1 + 2 * padding;
		nz = (int)ceil(nSeg[2]) + 1 + 2 * padding;
		hi = lo + Coord((Real)nx, (Real)ny, (Real)nz) * ds;

		num = nx * ny * nz;

		//		npMax = 128;

		cuSafeCall(cudaMalloc((void**)&counter, num * sizeof(int)));
		cuSafeCall(cudaMalloc((void**)&index, num * sizeof(int)));

		if (m_reduce != nullptr)
		{
			delete m_reduce;
		}

		m_reduce = Reduction<int>::Create(num);
	}

	template<typename TDataType>
	__global__ void K_CalculateParticleNumber(GridHash<TDataType> hash, DArray<typename TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		int gId = hash.getIndex(pos[pId]);

		if (gId != INVALID)
			atomicAdd(&(hash.index[gId]), 1);
	}

	template<typename TDataType>
	__global__ void K_ConstructHashTable(GridHash<TDataType> hash, DArray<typename TDataType::Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		int gId = hash.getIndex(pos[pId]);

		if (gId < 0) return;

		int index = atomicAdd(&(hash.counter[gId]), 1);
		hash.ids[hash.index[gId] + index] = pId;
	}

	template<typename TDataType>
	void GridHash<TDataType>::construct(DArray<Coord>& pos)
	{
		clear();

		dim3 pDims = int(ceil(pos.size() / BLOCK_SIZE + 0.5f));

		K_CalculateParticleNumber << <pDims, BLOCK_SIZE >> > (*this, pos);
		particle_num = m_reduce->accumulate(index, num);

		if (m_scan == nullptr)
		{
			m_scan = new Scan<int>();
		}
		m_scan->exclusive(index, num);

		if (ids != nullptr)
		{
			cuSafeCall(cudaFree(ids));
		}
		cuSafeCall(cudaMalloc((void**)&ids, particle_num * sizeof(int)));

		//		std::cout << "Particle number: " << particle_num << std::endl;

		K_ConstructHashTable << <pDims, BLOCK_SIZE >> > (*this, pos);
		cuSynchronize();
	}

	template<typename TDataType>
	void GridHash<TDataType>::clear()
	{
		if (counter != nullptr)
			cuSafeCall(cudaMemset(counter, 0, num * sizeof(int)));
		
		if (index != nullptr)
			cuSafeCall(cudaMemset(index, 0, num * sizeof(int)));
	}

	template<typename TDataType>
	void GridHash<TDataType>::release()
	{
		if (counter != nullptr)
			cuSafeCall(cudaFree(counter));

		if (ids != nullptr)
			cuSafeCall(cudaFree(ids));

		if (index != nullptr)
			cuSafeCall(cudaFree(index));

		if (m_scan != nullptr)
			delete m_scan;

		if (m_reduce != nullptr)
			delete m_reduce;
	}

	DEFINE_CLASS(GridHash);
}