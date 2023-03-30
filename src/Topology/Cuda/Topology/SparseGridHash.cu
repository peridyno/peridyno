#include "SparseGridHash.h"

#include "Object.h"
#include "DataTypes.h"

#include <thrust/sort.h>

namespace dyno 
{
	template<typename TDataType>
	SparseGridHash<TDataType>::SparseGridHash()
	{
	}

	template<typename TDataType>
	SparseGridHash<TDataType>::~SparseGridHash()
	{
	}

	template<typename TDataType>
	void SparseGridHash<TDataType>::setSpace(Coord lo, Real h, Real L)
	{
		m_lo = lo;
		m_h = h;
		m_L = L;

		Real segments = m_L / h;

		m_level_max = ceil(log2(segments));
	}

	template<typename Coord>
	DYN_FUNC void CalculateIndex(
		int& i,
		int& j,
		int& k,
		Coord p, 
		Coord origin, 
		Real L, 
		int level)
	{
		int grid_size = (int)pow(Real(2), int(level));

		Coord p_rel = p - origin;

		i = (int)floor(p_rel.x / L * grid_size);
		j = (int)floor(p_rel.y / L * grid_size);
		k = (int)floor(p_rel.z / L * grid_size);
	}

	template<typename Coord>
	__global__ void SGH_CreateAllNodes(
		DArray<OctreeNode> nodes,
		DArray<Coord> points,
		Coord origin,
		Real L,
		int level)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= nodes.size()) return;

		int i, j, k;
		CalculateIndex(i, j, k, points[tId], origin, L, level);

		nodes[tId] = OctreeNode(level, i, j, k);

		nodes[tId].setDataIndex(tId);
	}

	template<typename TDataType>
	void SparseGridHash<TDataType>::construct(DArray<Coord>& points, Real h)
	{
		m_h = h;

		Reduction<Coord> reduce;
		Coord maxV = reduce.maximum(points.begin(), points.size());
		Coord minV = reduce.minimum(points.begin(), points.size());

		Real maxL = maximum(abs(maxV.x - minV.x), maximum(abs(maxV.y - minV.y), abs(maxV.z - minV.z)));

		Real segments = m_L / h;
		m_level_max = std::max(ceil(log2(segments)), Real(2));

		m_L = m_h * pow(Real(2), m_level_max);

		m_lo = (maxV + minV) / 2 - m_L / 2;

		m_all_nodes.resize(points.size());

		cuExecute(m_all_nodes.size(),
			SGH_CreateAllNodes,
			m_all_nodes,
			points,
			m_lo,
			m_L,
			m_level_max);

		thrust::sort(thrust::device, m_all_nodes.begin(), m_all_nodes.begin() + m_all_nodes.size(), NodeCmp());
	}

	DEFINE_CLASS(SparseGridHash);
}