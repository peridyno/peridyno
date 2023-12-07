#include "ParticleSystemHelper.h"

#include "Algorithm/Reduction.h"

#include <thrust/sort.h>

namespace dyno
{
	__device__ OcKey K_CalculateMortonCode(OcIndex x, OcIndex y, OcIndex z)
	{
		OcKey key = 0;

		for (int i = 0; i < MAX_LEVEL; i++)
		{
			key |= (x & 1U << i) << 2 * i | (y & 1U << i) << (2 * i + 1) | (z & 1U << i) << (2 * i + 2);
		}
		return key;
	}

	template<typename Real, typename Coord>
	__global__ void PSH_CalculateMortonCode(
		DArray<OcKey> morton,
		DArray<Coord> pos,
		Coord lo,
		Coord hi,
		Real d)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		Coord p_i = pos[pId];
		OcIndex x = OcIndex((p_i.x - lo.x) / d);
		OcIndex y = OcIndex((p_i.y - lo.y) / d);
		OcIndex z = OcIndex((p_i.z - lo.z) / d);

		morton[pId] = K_CalculateMortonCode(x, y, z);
	}

	template<typename TDataType>
	void ParticleSystemHelper<TDataType>::calculateMortonCode(
		DArray<OcKey>& morton,
		DArray<Coord>& pos,
		Real d)
	{
		Reduction<Coord> reduce;
		Coord lo = reduce.minimum(pos.begin(), pos.size());
		Coord hi = reduce.maximum(pos.begin(), pos.size());

		cuExecute(pos.size(),
			PSH_CalculateMortonCode,
			morton,
			pos,
			lo,
			hi,
			d);
	}

	__global__ void PSH_InitParticleIds(
		DArray<uint> ids)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= ids.size()) return;

		ids[pId] = pId;
	}

	template<typename Coord>
	__global__ void PSH_ReorderParticles(
		DArray<Coord> target,
		DArray<Coord> source,
		DArray<uint> idsInOrder)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= idsInOrder.size()) return;

		target[pId] = source[idsInOrder[pId]];
	}

	template<typename TDataType>
	void ParticleSystemHelper<TDataType>::reorderParticles(
		DArray<Coord>& pos, 
		DArray<Coord>& vel, 
		DArray<Coord>& force,
		DArray<OcKey>& morton)
	{
		DArray<uint> idsInOrder(pos.size());

		cuExecute(pos.size(),
			PSH_InitParticleIds,
			idsInOrder);

		thrust::sort_by_key(thrust::device, morton.begin(), morton.begin() + morton.size(), idsInOrder.begin());

		DArray<Coord> buffer(pos.size());
		buffer.assign(pos);

		cuExecute(pos.size(),
			PSH_ReorderParticles,
			pos,
			buffer,
			idsInOrder);

		buffer.assign(vel);

		cuExecute(vel.size(),
			PSH_ReorderParticles,
			vel,
			buffer,
			idsInOrder);

		buffer.assign(force);
		cuExecute(force.size(),
			PSH_ReorderParticles,
			force,
			buffer,
			idsInOrder);

		idsInOrder.clear();
		buffer.clear();
	}

	template class ParticleSystemHelper<DataType3f>;
}