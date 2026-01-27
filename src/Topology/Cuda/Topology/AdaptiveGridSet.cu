#include "AdaptiveGridSet.h"
#include "Algorithm/Reduction.h"
#include "Array/ArrayMap.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <thrust/sort.h>
#include "Timer.h"

namespace dyno
{
	IMPLEMENT_TCLASS(AdaptiveGridSet, TDataType)

	//the mapping relation: index=(x+1)+(y+1)*3+(z+1)*9
	__constant__ int offset26[26][3] = {
			-1,-1,-1,
			0, -1, -1,
			1, -1, -1,
			-1, 0, -1,
			0, 0, -1,
			1, 0, -1,
			-1, 1, -1,
			0, 1, -1,
			1, 1, -1,
			-1, -1, 0,
			0, -1, 0,
			1, -1, 0,
			-1, 0, 0,
			1, 0, 0,
			-1, 1, 0,
			0, 1, 0,
			1, 1, 0,
			-1, -1, 1,
			0, -1, 1,
			1, -1, 1,
			-1, 0, 1,
			0, 0, 1,
			1, 0, 1,
			-1, 1, 1,
			0, 1, 1,
			1, 1, 1
	};

	__constant__ int offset6[6][3] = {
	-1,0,0,
	1, 0, 0,
	0, -1, 0,
	0, 1, 0,
	0, 0, -1,
	0, 0, 1
	};

	template<typename TDataType>
	AdaptiveGridSet<TDataType>::AdaptiveGridSet()
		: TopologyModule()
	{
	}

	template<typename TDataType>
	AdaptiveGridSet<TDataType>::~AdaptiveGridSet()
	{
	}

	template<typename TDataType>
	void AdaptiveGridSet<TDataType>::setNodes(DArray<AdaptiveGridNode>& nodes)
	{
		m_octree.resize(nodes.size());
		m_octree.assign(nodes);
	}

	template<typename TDataType>
	void AdaptiveGridSet<TDataType>::setNeighbors(DArrayList<int>& nodes)
	{
		m_neighbors.assign(nodes);
	}

	template <typename Coord>
	__global__ void AGS_ComputeLeafsAndNeighbors27(
		DArray<Coord> leafs_pos,
		DArray<uint> neighbors_count,
		DArray<int> count,
		DArray<AdaptiveGridNode> octree,
		DArrayList<int> neighbors)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		if (octree[tId].isLeaf())
		{
			leafs_pos[count[tId]] = octree[tId].m_position;

			neighbors_count[count[tId]] = neighbors[tId].size();
		}
	}

	__global__ void AGS_ComputeNeighbors27(
		DArrayList<int> leafs_neighbors,
		DArray<int> count,
		DArray<AdaptiveGridNode> octree,
		DArrayList<int> neighbors)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		if (octree[tId].isLeaf())
		{
			for (int i = 0; i < neighbors[tId].size(); i++)
				leafs_neighbors[count[tId]].insert(count[(neighbors[tId][i])]);
		}
	}

	template<typename TDataType>
	void AdaptiveGridSet<TDataType>::extractLeafs27(DArray<Coord>& pos, DArrayList<int>& neighbors)
	{
		if (m_neighbors27.size() != m_octree.size())
		{
			printf("Please make sure it's 27 neighbors? \n");
			return;
		}

		pos.resize(m_leafs_num);
		DArray<uint> count(m_leafs_num);
		cuExecute(m_leafIndex.size(),
			AGS_ComputeLeafsAndNeighbors27,
			pos,
			count,
			m_leafIndex,
			m_octree,
			m_neighbors27);

		neighbors.resize(count);
		count.clear();
		cuExecute(m_leafIndex.size(),
			AGS_ComputeNeighbors27,
			neighbors,
			m_leafIndex,
			m_octree,
			m_neighbors27);
	}

	__global__ void AGS_ComputeLeafsAndNeighbors6(
		DArray<AdaptiveGridNode> leafs,
		DArray<uint> neighbors_count,
		DArray<int> leaf_index,
		DArray<AdaptiveGridNode> octree,
		DArrayList<int> neighbors)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= leaf_index.size()) return;

		if (octree[tId].isLeaf())
		{
			leafs[leaf_index[tId]] = octree[tId];

			neighbors_count[6 * leaf_index[tId]] = neighbors[6 * tId].size();
			neighbors_count[6 * leaf_index[tId] + 1] = neighbors[6 * tId + 1].size();
			neighbors_count[6 * leaf_index[tId] + 2] = neighbors[6 * tId + 2].size();
			neighbors_count[6 * leaf_index[tId] + 3] = neighbors[6 * tId + 3].size();
			neighbors_count[6 * leaf_index[tId] + 4] = neighbors[6 * tId + 4].size();
			neighbors_count[6 * leaf_index[tId] + 5] = neighbors[6 * tId + 5].size();
		}
	}
	__global__ void AGS_ComputeNeighbors6(
		DArrayList<int> leafs_neighbors,
		DArray<int> count,
		DArray<AdaptiveGridNode> octree,
		DArrayList<int> neighbors)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		if (octree[tId].isLeaf())
		{
			for (int i = 0; i < neighbors[6 * tId].size(); i++)
				leafs_neighbors[6 * count[tId]].insert(count[(neighbors[6 * tId][i])]);
			for (int i = 0; i < neighbors[6 * tId + 1].size(); i++)
				leafs_neighbors[6 * count[tId] + 1].insert(count[(neighbors[6 * tId + 1][i])]);
			for (int i = 0; i < neighbors[6 * tId + 2].size(); i++)
				leafs_neighbors[6 * count[tId] + 2].insert(count[(neighbors[6 * tId + 2][i])]);
			for (int i = 0; i < neighbors[6 * tId + 3].size(); i++)
				leafs_neighbors[6 * count[tId] + 3].insert(count[(neighbors[6 * tId + 3][i])]);
			for (int i = 0; i < neighbors[6 * tId + 4].size(); i++)
				leafs_neighbors[6 * count[tId] + 4].insert(count[(neighbors[6 * tId + 4][i])]);
			for (int i = 0; i < neighbors[6 * tId + 5].size(); i++)
				leafs_neighbors[6 * count[tId] + 5].insert(count[(neighbors[6 * tId + 5][i])]);
		}
	}
	template<typename TDataType>
	void AdaptiveGridSet<TDataType>::extractLeafs6(DArray<AdaptiveGridNode>& leafs, DArrayList<int>& neighbors)
	{
		if (m_neighbors.size() != 6 * m_octree.size())
		{
			printf("Please make sure it's 6 neighbors? \n");
			return;
		}

		leafs.resize(m_leafs_num);
		DArray<uint> count(6 * m_leafs_num);
		cuExecute(m_leafIndex.size(),
			AGS_ComputeLeafsAndNeighbors6,
			leafs,
			count,
			m_leafIndex,
			m_octree,
			m_neighbors);

		neighbors.resize(count);
		count.clear();
		cuExecute(m_leafIndex.size(),
			AGS_ComputeNeighbors6,
			neighbors,
			m_leafIndex,
			m_octree,
			m_neighbors);
	}

	__global__ void AGS_ComputeLeafs(
		DArray<AdaptiveGridNode> leafs,
		DArray<AdaptiveGridNode> octree,
		DArray<int> leaf_index)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= leaf_index.size()) return;

		if (octree[tId].isLeaf())
			leafs[leaf_index[tId]] = octree[tId];
	}
	template<typename TDataType>
	void AdaptiveGridSet<TDataType>::extractLeafs(DArray<AdaptiveGridNode>& leafs)
	{
		leafs.resize(m_leafs_num);
		cuExecute(m_leafIndex.size(),
			AGS_ComputeLeafs,
			leafs,
			m_octree,
			m_leafIndex);
	}


	template <typename Real>
	__global__ void AGS_CountLeafs2D(
		DArray<int> leafs,
		DArray<AdaptiveGridNode> octree,
		DArrayList<int> neighbors,
		Real dx,
		Level max_level,
		Real zpos)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= leafs.size()) return;

		if (octree[tId].isLeaf())
		{
			Real up_dx = dx * (1 << (max_level - (octree[tId].m_level)));

			if (((octree[tId].m_position[2] - 0.5*up_dx) <= zpos) && ((octree[tId].m_position[2] + 0.5*up_dx)+ REAL_EPSILON >= zpos))
				leafs[tId] = 1;
		}
	}

	__global__ void AGS_ComputeLeafsAndNeighbors2D(
		DArray<AdaptiveGridNode> leafs_pos,
		DArray<uint> neighbors_count,
		DArray<int> count,
		DArray<AdaptiveGridNode> octree,
		DArrayList<int> neighbors)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		if ((tId == (count.size() - 1) && count[tId] < leafs_pos.size()) )
		{
			leafs_pos[count[tId]] = octree[tId];

			neighbors_count[4 * count[tId]] = std::min((int)neighbors[6 * tId].size(), 2);
			neighbors_count[4 * count[tId] + 1] = std::min((int)neighbors[6 * tId + 1].size(), 2);
			neighbors_count[4 * count[tId] + 2] = std::min((int)neighbors[6 * tId + 2].size(), 2);
			neighbors_count[4 * count[tId] + 3] = std::min((int)neighbors[6 * tId + 3].size(), 2);
		}
		else if (count[tId] < count[tId + 1])
		{
			leafs_pos[count[tId]] = octree[tId];

			neighbors_count[4 * count[tId]] = std::min((int)neighbors[6 * tId].size(), 2);
			neighbors_count[4 * count[tId] + 1] = std::min((int)neighbors[6 * tId + 1].size(), 2);
			neighbors_count[4 * count[tId] + 2] = std::min((int)neighbors[6 * tId + 2].size(), 2);
			neighbors_count[4 * count[tId] + 3] = std::min((int)neighbors[6 * tId + 3].size(), 2);
		}
	}

	__global__ void AGS_ComputeNeighbors2D(
		DArrayList<int> leafs_neighbors,
		DArray<int> count,
		DArray<int> cindex,
		DArray<AdaptiveGridNode> octree,
		DArrayList<int> neighbors,
		int leafs_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		if (cindex[tId] == 1)
		{
			for (int i = 0; i < neighbors[6 * tId].size(); i++)
			{
				if (cindex[(neighbors[6 * tId][i])] == 1)
					leafs_neighbors[4 * count[tId]].insert(count[(neighbors[6 * tId][i])]);
			}
			for (int i = 0; i < neighbors[6 * tId + 1].size() && i < 2; i++)
			{
				if (cindex[(neighbors[6 * tId + 1][i])] == 1)
					leafs_neighbors[4 * count[tId] + 1].insert(count[(neighbors[6 * tId + 1][i])]);
			}
			for (int i = 0; i < neighbors[6 * tId + 2].size() && i < 2; i++)
			{
				if (cindex[(neighbors[6 * tId + 2][i])] == 1)
					leafs_neighbors[4 * count[tId] + 2].insert(count[(neighbors[6 * tId + 2][i])]);
			}
			for (int i = 0; i < neighbors[6 * tId + 3].size() && i < 2; i++)
			{
				if (cindex[(neighbors[6 * tId + 3][i])] == 1)
					leafs_neighbors[4 * count[tId] + 3].insert(count[(neighbors[6 * tId + 3][i])]);
			}
		}
	}

	template<typename TDataType>
	void AdaptiveGridSet<TDataType>::extractLeafs2D(DArray<AdaptiveGridNode>& pos, DArrayList<int>& neighbors, Real zAxis)
	{
		DArray<int> m_leafIndex2D(m_octree.size());
		m_leafIndex2D.reset();
		//count the number of leaf nodes
		cuExecute(m_leafIndex2D.size(),
			AGS_CountLeafs2D,
			m_leafIndex2D,
			m_octree,
			m_neighbors,
			m_dx,
			m_level_max,
			zAxis);

		int leafs_num = thrust::reduce(thrust::device, m_leafIndex2D.begin(), m_leafIndex2D.begin() + m_leafIndex2D.size(), (int)0, thrust::plus<int>());
		std::printf("GetLeafs2D: the number of all nodes and leafs are: %d %d \n", m_octree.size(), leafs_num);
		if (leafs_num == 0)
		{
			pos.resize(0);
			return;
		}

		DArray<int> leafIndex2D;
		leafIndex2D.assign(m_leafIndex2D);
		thrust::exclusive_scan(thrust::device, m_leafIndex2D.begin(), m_leafIndex2D.begin() + m_leafIndex2D.size(), m_leafIndex2D.begin());

		pos.resize(leafs_num);
		DArray<uint> neighbors_count(4 * leafs_num);
		neighbors_count.reset();
		//count the number of neighbors
		cuExecute(m_leafIndex2D.size(),
			AGS_ComputeLeafsAndNeighbors2D,
			pos,
			neighbors_count,
			m_leafIndex2D,
			m_octree,
			m_neighbors);

		neighbors.resize(neighbors_count);
		//compute the neighbors
		cuExecute(m_leafIndex2D.size(),
			AGS_ComputeNeighbors2D,
			neighbors,
			m_leafIndex2D,
			leafIndex2D,
			m_octree,
			m_neighbors,
			leafs_num);

		neighbors_count.clear();
		leafIndex2D.clear();
		m_leafIndex2D.clear();
	}

	template <typename Real, typename Coord>
	__global__ void AGS_Assess(
		DArray<int> index,
		DArray<Coord> pos,
		DArray<AdaptiveGridNode> octree,
		DArray<int> leafIndex,
		Coord m_origin,
		Real m_dx,
		Level m_level)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size()) return;

		Coord resid = (pos[tId] - m_origin) / m_dx;
		int i = (int)floor(resid[0]);
		int j = (int)floor(resid[1]);
		int k = (int)floor(resid[2]);
		int resolution = (1 << m_level);
		if ((i < 0 || i >= resolution) || (j < 0 || j >= resolution) || (k < 0 || k >= resolution))
		{
			index[tId] = EMPTY;
			return;
		}

		OcKey morton = CalculateMortonCode(i, j, k);
		Level l0 = 1;
		OcKey morton_l0 = morton >> (3 * (m_level - l0));
		int ind = morton_l0 & 7;		
		while (!octree[ind].isLeaf())
		{
			ind = octree[ind].m_fchild;

			l0++;
			morton_l0 = morton >> (3 * (m_level - l0));
			ind += morton_l0 & 7;
		}

		index[tId] = leafIndex[ind];
	}

	template<typename TDataType>
	void AdaptiveGridSet<TDataType>::accessRandom(DArray<int>& index, DArray<Coord>& pos)
	{
		if (index.size() != pos.size())
			index.resize(pos.size());

		cuExecute(pos.size(),
			AGS_Assess,
			index,
			pos,
			m_octree,
			m_leafIndex,
			m_origin,
			m_dx,
			m_level_max);
	}

	__global__ void AGS_CountLeafs(
		DArray<int> leafs,
		DArray<AdaptiveGridNode> octree)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= leafs.size()) return;

		if (octree[tId].isLeaf())
			leafs[tId] = 1;
	}
	template<typename TDataType>
	void AdaptiveGridSet<TDataType>::ConstructLeafs()
	{
		m_leafIndex.resize(m_octree.size());
		m_leafIndex.reset();
		//count the number of leaf nodes
		cuExecute(m_leafIndex.size(),
			AGS_CountLeafs,
			m_leafIndex,
			m_octree);

		m_leafs_num = thrust::reduce(thrust::device, m_leafIndex.begin(), m_leafIndex.begin() + m_leafIndex.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, m_leafIndex.begin(), m_leafIndex.begin() + m_leafIndex.size(), m_leafIndex.begin());
		std::printf("GetLeafs: the number of leafs is: %d  %d;  %hu  %hu \n", m_leafs_num, m_octree.size(), m_level_max, m_level_num);
	}

	template<typename TDataType>
	void AdaptiveGridSet<TDataType>::updateTopology()
	{
		ConstructLeafs();

		ConstructNeighborsSix();

		if (m_neighbor_type == 1)
			ConstructNeighborsTwentySeven();

		//ConstructVertex();
	}

	template<typename TDataType>
	__global__ void AGS_CountNeighborhoodForest(
		DArray<uint> ncount,
		DArray<AdaptiveGridNode> nodes,
		Level levelmin,
		AdaptiveGridSet<TDataType> gridSet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		auto alpha = [&](OcIndex i, OcIndex j, OcIndex k, Level gl) -> bool {
			int gresolution = (1 << gl);
			if (i < 0 || i >= gresolution)
				return false;
			if (j < 0 || j >= gresolution)
				return false;
			if (k < 0 || k >= gresolution)
				return false;

			return true;
			};

		if (nodes[tId].isLeaf())
		{
			Level gl = nodes[tId].m_level;
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)nodes[tId].m_morton, gnx, gny, gnz);

			for (int c = 0; c < 6; c++)
			{
				if (alpha(gnx + offset6[c][0], gny + offset6[c][1], gnz + offset6[c][2], gl))
				{
					int nindex = 0;
					OcKey nmorton = CalculateMortonCode(gnx + offset6[c][0], gny + offset6[c][1], gnz + offset6[c][2]);
					if (gridSet.accessRandom(nindex, nmorton, gl))
					{
						ncount[tId * 6 + c] = 1;
						if (gl > (nodes[nindex].m_level))
						{
							int nc = c + 1 - 2 * (c % 2);
							ncount[nindex * 6 + nc] = 4;
						}
					}
				}
			}
		}
	}

	template<typename TDataType>
	__global__ void AGS_ComputeNeighborhoodForest(
		DArrayList<int> neighbors,
		DArray<AdaptiveGridNode> nodes,
		Level levelmin,
		AdaptiveGridSet<TDataType> gridSet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		if (nodes[tId].isLeaf())
		{
			Level gl = nodes[tId].m_level;
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)nodes[tId].m_morton, gnx, gny, gnz);

			int ind = tId - (nodes[tId].m_morton & 7);
			int gresolution = (1 << gl);
			int nindex;
			if (gnx > 0)
			{
				OcKey nmorton = CalculateMortonCode(gnx - 1, gny, gnz);
				if (gridSet.accessRandom(nindex, nmorton, gl))
				{
					neighbors[tId * 6].insert(nindex);

					if (gl > (nodes[nindex].m_level))
					{
						neighbors[nindex * 6 + 1].insert(ind);
						neighbors[nindex * 6 + 1].insert(ind + 2);
						neighbors[nindex * 6 + 1].insert(ind + 4);
						neighbors[nindex * 6 + 1].insert(ind + 6);
					}
				}
			}
			if (gnx < (gresolution - 1))
			{
				OcKey nmorton = CalculateMortonCode(gnx + 1, gny, gnz);
				if (gridSet.accessRandom(nindex, nmorton, gl))
				{
					neighbors[tId * 6 + 1].insert(nindex);

					if (gl > (nodes[nindex].m_level))
					{
						neighbors[nindex * 6].insert(ind + 1);
						neighbors[nindex * 6].insert(ind + 3);
						neighbors[nindex * 6].insert(ind + 5);
						neighbors[nindex * 6].insert(ind + 7);
					}
				}
			}
			if (gny > 0)
			{
				OcKey nmorton = CalculateMortonCode(gnx, gny - 1, gnz);
				if (gridSet.accessRandom(nindex, nmorton, gl))
				{
					neighbors[tId * 6 + 2].insert(nindex);

					if (gl > (nodes[nindex].m_level))
					{
						neighbors[nindex * 6 + 3].insert(ind);
						neighbors[nindex * 6 + 3].insert(ind + 1);
						neighbors[nindex * 6 + 3].insert(ind + 4);
						neighbors[nindex * 6 + 3].insert(ind + 5);
					}
				}
			}
			if (gny < (gresolution - 1))
			{
				OcKey nmorton = CalculateMortonCode(gnx, gny + 1, gnz);
				if (gridSet.accessRandom(nindex, nmorton, gl))
				{
					neighbors[tId * 6 + 3].insert(nindex);

					if (gl > (nodes[nindex].m_level))
					{
						neighbors[nindex * 6 + 2].insert(ind + 2);
						neighbors[nindex * 6 + 2].insert(ind + 3);
						neighbors[nindex * 6 + 2].insert(ind + 6);
						neighbors[nindex * 6 + 2].insert(ind + 7);
					}
				}
			}
			if (gnz > 0)
			{
				OcKey nmorton = CalculateMortonCode(gnx, gny, gnz - 1);
				if (gridSet.accessRandom(nindex, nmorton, gl))
				{
					neighbors[tId * 6 + 4].insert(nindex);

					if (gl > (nodes[nindex].m_level))
					{
						neighbors[nindex * 6 + 5].insert(ind);
						neighbors[nindex * 6 + 5].insert(ind + 1);
						neighbors[nindex * 6 + 5].insert(ind + 2);
						neighbors[nindex * 6 + 5].insert(ind + 3);
					}
				}
			}
			if (gnz < (gresolution - 1))
			{
				OcKey nmorton = CalculateMortonCode(gnx, gny, gnz + 1);
				if (gridSet.accessRandom(nindex, nmorton, gl))
				{
					neighbors[tId * 6 + 5].insert(nindex);

					if (gl > (nodes[nindex].m_level))
					{
						neighbors[nindex * 6 + 4].insert(ind + 4);
						neighbors[nindex * 6 + 4].insert(ind + 5);
						neighbors[nindex * 6 + 4].insert(ind + 6);
						neighbors[nindex * 6 + 4].insert(ind + 7);
					}
				}
			}
		}
	}

	template<typename TDataType>
	void AdaptiveGridSet<TDataType>::ConstructNeighborsSix()
	{
		int node_num = m_octree.size();
		DArray<uint> data_count(node_num * 6);
		data_count.reset();
		cuExecute(node_num,
			AGS_CountNeighborhoodForest,
			data_count,
			m_octree,
			m_level_max - m_level_num + 1,
			*this);

		m_neighbors.resize(data_count);
		cuExecute(node_num,
			AGS_ComputeNeighborhoodForest,
			m_neighbors,
			m_octree,
			m_level_max - m_level_num + 1,
			*this);
		data_count.clear();
	}
	
	template <typename TDataType>
	__global__ void AGS_ComputeAllNeighborhood27(
		DArrayList<int> ncount,
		DArray<AdaptiveGridNode> nodes,
		int node_num,
		Level levelmin,
		AdaptiveGridSet<TDataType> gridSet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= node_num) return;

		if (nodes[tId].isLeaf())
		{
			Level gl = nodes[tId].m_level;
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)nodes[tId].m_morton, gnx, gny, gnz);
			int gresolution = (1 << gl);

			auto alpha = [&](OcIndex i, OcIndex j, OcIndex k) -> bool {
				if (i < 0 || i >= gresolution)
					return false;
				if (j < 0 || j >= gresolution)
					return false;
				if (k < 0 || k >= gresolution)
					return false;

				return true;
				};

			int nindex = 0;
			for (int c = 0; c < 27; c++)
			{
				if (alpha(gnx + offset26[c][0], gny + offset26[c][1], gnz + offset26[c][2]))
				{
					OcKey nmorton = CalculateMortonCode(gnx + offset26[c][0], gny + offset26[c][1], gnz + offset26[c][2]);
					if (gridSet.accessRandom(nindex, nmorton, gl))
					{
						ncount[tId].atomicInsert(nindex);
						if (gl > (nodes[nindex].m_level))
							ncount[nindex].atomicInsert(tId);
					}
				}
			}
		}
	}
	
	__global__ void AGS_ComputeNoRepeatNeighborhood27(
		DArrayMap<bool> nmap,
		DArrayList<int> nlist,
		DArray<AdaptiveGridNode> nodes,
		int node_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= node_num) return;

		if (nodes[tId].isLeaf())
		{
			for (int i = 0; i < nlist[tId].size(); i++)
				nmap[tId].insert(Pair<int, bool>(nlist[tId][i], true));
		}
	}

	__global__ void AGS_CountNeighborhood27(
		DArray<uint> count,
		DArrayMap<bool> nmap,
		DArray<AdaptiveGridNode> nodes,
		int node_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= node_num) return;

		if (nodes[tId].isLeaf())
			count[tId] = nmap[tId].size();
	}

	__global__ void AGS_ComputeNeighborhood27(
		DArrayList<int> nlist,
		DArrayMap<bool> nmap,
		DArray<AdaptiveGridNode> nodes,
		int node_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= node_num) return;

		if (nodes[tId].isLeaf())
		{
			for (int i = 0; i < nmap[tId].size(); i++)
				nlist[tId].insert(nmap[tId][i].first);
		}
	}

	template<typename TDataType>
	void AdaptiveGridSet<TDataType>::ConstructNeighborsTwentySeven()
	{
		int node_num = m_octree.size();
		DArrayList<int> alln;
		alln.resize(node_num, 56);
		cuExecute(node_num,
			AGS_ComputeAllNeighborhood27,
			alln,
			m_octree,
			node_num,
			m_level_max - m_level_num + 1,
			*this);

		DArrayMap<bool> neighbor0;
		neighbor0.resize(node_num, 56);
		neighbor0.reset();
		cuExecute(node_num,
			AGS_ComputeNoRepeatNeighborhood27,
			neighbor0,
			alln,
			m_octree,
			node_num);
		alln.clear();

		DArray<uint> data_count(node_num);
		data_count.reset();
		cuExecute(node_num,
			AGS_CountNeighborhood27,
			data_count,
			neighbor0,
			m_octree,
			node_num);

		m_neighbors27.resize(data_count);
		data_count.clear();
		cuExecute(node_num,
			AGS_ComputeNeighborhood27,
			m_neighbors27,
			neighbor0,
			m_octree,
			node_num);
		neighbor0.clear();
	}

	template<typename Real>
	__global__ void AGS_ComputeAllVertex(
		DArray<OcKey> vertex,
		DArray<AdaptiveGridNode> nodes,
		DArray<int> index,
		Level lmax,
		Real dx)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		if (nodes[tId].isLeaf())
		{
			OcIndex nx, ny, nz;
			RecoverFromMortonCode(nodes[tId].m_morton, nx, ny, nz);

			vertex[8 * index[tId]] = (nodes[tId].m_morton) << (3 * (lmax - nodes[tId].m_level));//(-1,-1,-1)
			vertex[8 * index[tId] + 1] = (CalculateMortonCode(nx + 1, ny, nz)) << (3 * (lmax - nodes[tId].m_level));//(1,-1,-1)
			vertex[8 * index[tId] + 2] = (CalculateMortonCode(nx + 1, ny + 1, nz)) << (3 * (lmax - nodes[tId].m_level));//(1,1,-1)
			vertex[8 * index[tId] + 3] = (CalculateMortonCode(nx, ny + 1, nz)) << (3 * (lmax - nodes[tId].m_level));//(-1,1,-1)
			vertex[8 * index[tId] + 4] = (CalculateMortonCode(nx, ny, nz + 1)) << (3 * (lmax - nodes[tId].m_level));//(-1,-1,1)
			vertex[8 * index[tId] + 5] = (CalculateMortonCode(nx + 1, ny, nz + 1)) << (3 * (lmax - nodes[tId].m_level));//(1,-1,1)
			vertex[8 * index[tId] + 6] = (CalculateMortonCode(nx + 1, ny + 1, nz + 1)) << (3 * (lmax - nodes[tId].m_level));//(1,1,1)
			vertex[8 * index[tId] + 7] = (CalculateMortonCode(nx, ny + 1, nz + 1)) << (3 * (lmax - nodes[tId].m_level));//(-1,1,1)
		}
	}

	__global__ void AGS_CountNoRepeatVertex(
		DArray<int> count,
		DArray<OcKey> vertex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertex.size()) return;

		if (tId == 0 || vertex[tId] != vertex[tId - 1])
			count[tId] = 1;
	}

	template <typename Real, typename Coord, typename TDataType>
	__global__ void AGS_ComputeVertexIndex(
		DArray<int> node2vertex,
		DArray<Coord> vertex_pos,
		DArray<OcKey> vertex,
		DArray<int> vcount,
		DArray<AdaptiveGridNode> nodes,
		DArray<int> leaf_index,
		Real dx,
		Coord origin,
		Level lmin,
		Level lmax,
		int resolution,
		AdaptiveGridSet<TDataType> gridSet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertex.size()) return;

		if (tId == 0 || vertex[tId] != vertex[tId - 1])
		{
			OcIndex nx, ny, nz;
			RecoverFromMortonCode(vertex[tId], nx, ny, nz);

			auto alpha = [&](int dnx_, int dny_, int dnz_, int& index_, Coord& resid_) -> bool {
				if ((nx + dnx_) < 0 || (nx + dnx_) >= resolution) return false;
				if ((ny + dny_) < 0 || (ny + dny_) >= resolution) return false;
				if ((nz + dnz_) < 0 || (nz + dnz_) >= resolution) return false;
				gridSet.accessRandom(index_, CalculateMortonCode((nx + dnx_), (ny + dny_), (nz + dnz_)), lmax);
				Real up_dx = dx * (resolution >> (nodes[index_].m_level));
				resid_ = nodes[index_].m_position - Coord((0.5 + dnx_) * up_dx, (0.5 + dny_) * up_dx, (0.5 + dnz_) * up_dx);
				return true;
				};

			Coord vpos = origin + Coord(nx * dx, ny * dx, nz * dx);
			vertex_pos[vcount[tId]] = vpos;
			int n_index;
			Coord resid;
			if (alpha(0, 0, 0, n_index, resid) && ((resid - vpos).norm() < 0.1 * dx))
				node2vertex[8 * leaf_index[n_index]] = vcount[tId];
			if (alpha(-1, 0, 0, n_index, resid) && ((resid - vpos).norm() < 0.1 * dx))
				node2vertex[8 * leaf_index[n_index] + 1] = vcount[tId];
			if (alpha(-1, -1, 0, n_index, resid) && ((resid - vpos).norm() < 0.1 * dx))
				node2vertex[8 * leaf_index[n_index] + 2] = vcount[tId];
			if (alpha(0, -1, 0, n_index, resid) && ((resid - vpos).norm() < 0.1 * dx))
				node2vertex[8 * leaf_index[n_index] + 3] = vcount[tId];
			if (alpha(0, 0, -1, n_index, resid) && ((resid - vpos).norm() < 0.1 * dx))
				node2vertex[8 * leaf_index[n_index] + 4] = vcount[tId];
			if (alpha(-1, 0, -1, n_index, resid) && ((resid - vpos).norm() < 0.1 * dx))
				node2vertex[8 * leaf_index[n_index] + 5] = vcount[tId];
			if (alpha(-1, -1, -1, n_index, resid) && ((resid - vpos).norm() < 0.1 * dx))
				node2vertex[8 * leaf_index[n_index] + 6] = vcount[tId];
			if (alpha(0, -1, -1, n_index, resid) && ((resid - vpos).norm() < 0.1 * dx))
				node2vertex[8 * leaf_index[n_index] + 7] = vcount[tId];
		}
	}

	__global__ void AGS_InitVertexNeighbor(
		DArray<int> vneighbor)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vneighbor.size()) return;

		vneighbor[tId] = EMPTY;
	}

	__global__ void AGS_ComputeVertexNeighbor(
		DArray<int> vneighbor,
		DArray<int> node2vertex, 
		DArray<AdaptiveGridNode> nodes,
		DArray<int> leaf_index,
		DArrayList<int> neighbor)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		auto alpha = [&](int _index1, int _index2) -> bool {
			if (neighbor[6 * tId + _index1].size() == 0 || neighbor[6 * tId + _index2].size() == 0) return true;
			if ((nodes[neighbor[6 * tId + _index1][0]].m_level == nodes[neighbor[6 * tId + _index2][0]].m_level) && (nodes[neighbor[6 * (neighbor[6 * tId + _index1][0]) + _index2][0]].m_level > nodes[tId].m_level))
				return false;

			return true;
			};

		if (nodes[tId].isLeaf())
		{
			int lindex = leaf_index[tId];
			//neighbor order is: -x,+x,-y,+y,-z,+z
			//vertex order is:(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1);(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)
			int v1 = node2vertex[8 * lindex];
			int v2 = node2vertex[8 * lindex + 1];
			int v3 = node2vertex[8 * lindex + 2];
			int v4 = node2vertex[8 * lindex + 3];
			int v5 = node2vertex[8 * lindex + 4];
			int v6 = node2vertex[8 * lindex + 5];
			int v7 = node2vertex[8 * lindex + 6];
			int v8 = node2vertex[8 * lindex + 7];

			if (neighbor[6 * tId].size() <= 1 && neighbor[6 * tId + 2].size() <= 1)
			{//v1&v5
				if (alpha(0, 2))
				{
					vneighbor[6 * v1 + 5] = v5;
					vneighbor[6 * v5 + 4] = v1;
				}
			}
			if (neighbor[6 * tId].size() <= 1 && neighbor[6 * tId + 3].size() <= 1)
			{//v4&v8
				if (alpha(0, 3))
				{
					vneighbor[6 * v4 + 5] = v8;
					vneighbor[6 * v8 + 4] = v4;
				}
			}
			if (neighbor[6 * tId].size() <= 1 && neighbor[6 * tId + 4].size() <= 1)
			{//v1&v4
				if (alpha(0, 4))
				{
					vneighbor[6 * v1 + 3] = v4;
					vneighbor[6 * v4 + 2] = v1;
				}
			}
			if (neighbor[6 * tId].size() <= 1 && neighbor[6 * tId + 5].size() <= 1)
			{//v5&v8
				if (alpha(0, 5))
				{
					vneighbor[6 * v5 + 3] = v8;
					vneighbor[6 * v8 + 2] = v5;
				}
			}
			if (neighbor[6 * tId + 1].size() <= 1 && neighbor[6 * tId + 2].size() <= 1)
			{//v2&v6
				if (alpha(1, 2))
				{
					vneighbor[6 * v2 + 5] = v6;
					vneighbor[6 * v6 + 4] = v2;
				}
			}
			if (neighbor[6 * tId + 1].size() <= 1 && neighbor[6 * tId + 3].size() <= 1)
			{//v3&v7
				if (alpha(1, 3))
				{
					vneighbor[6 * v3 + 5] = v7;
					vneighbor[6 * v7 + 4] = v3;
				}
			}
			if (neighbor[6 * tId + 1].size() <= 1 && neighbor[6 * tId + 4].size() <= 1)
			{//v2&v3
				if (alpha(1, 4))
				{
					vneighbor[6 * v2 + 3] = v3;
					vneighbor[6 * v3 + 2] = v2;
				}
			}
			if (neighbor[6 * tId + 1].size() <= 1 && neighbor[6 * tId + 5].size() <= 1)
			{//v6&v7
				if (alpha(1, 5))
				{
					vneighbor[6 * v6 + 3] = v7;
					vneighbor[6 * v7 + 2] = v6;
				}
			}
			if (neighbor[6 * tId + 2].size() <= 1 && neighbor[6 * tId + 4].size() <= 1)
			{//v1&v2
				if (alpha(2, 4))
				{
					vneighbor[6 * v1 + 1] = v2;
					vneighbor[6 * v2 + 0] = v1;
				}
			}
			if (neighbor[6 * tId + 2].size() <= 1 && neighbor[6 * tId + 5].size() <= 1)
			{//v5&v6
				if (alpha(2, 5))
				{
					vneighbor[6 * v5 + 1] = v6;
					vneighbor[6 * v6 + 0] = v5;
				}
			}
			if (neighbor[6 * tId + 3].size() <= 1 && neighbor[6 * tId + 4].size() <= 1)
			{//v3&v4
				if (alpha(3, 4))
				{
					vneighbor[6 * v3 + 0] = v4;
					vneighbor[6 * v4 + 1] = v3;
				}
			}
			if (neighbor[6 * tId + 3].size() <= 1 && neighbor[6 * tId + 5].size() <= 1)
			{//v7&v8
				if (alpha(3, 5))
				{
					vneighbor[6 * v7 + 0] = v8;
					vneighbor[6 * v8 + 1] = v7;
				}
			}
		}
	}

	template<typename TDataType>
	void AdaptiveGridSet<TDataType>::extractVertex(DArray<Coord>& m_vertex, DArray<int>& m_vertex_neighbor, DArray<int>& m_node2Ver)
	{
		DArray<OcKey> vertex(8 * m_leafs_num);
		cuExecute(m_octree.size(),
			AGS_ComputeAllVertex,
			vertex,
			m_octree,
			m_leafIndex,
			m_level_max,
			m_dx);

		thrust::sort(thrust::device, vertex.begin(), vertex.begin() + vertex.size(), thrust::greater<OcKey>());

		DArray<int> count(vertex.size());
		count.reset();
		cuExecute(vertex.size(),
			AGS_CountNoRepeatVertex,
			count,
			vertex);

		int vertex_num = thrust::reduce(thrust::device, count.begin(), count.begin() + count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, count.begin(), count.begin() + count.size(), count.begin());

		int max_resolution = (1 << m_level_max);
		m_vertex.resize(vertex_num);
		m_node2Ver.resize(8 * m_leafs_num);
		cuExecute(vertex.size(),
			AGS_ComputeVertexIndex,
			m_node2Ver,
			m_vertex,
			vertex,
			count,
			m_octree,
			m_leafIndex,
			m_dx,
			m_origin,
			m_level_max - m_level_num + 1,
			m_level_max,
			max_resolution,
			*this);
		vertex.clear();
		count.clear();

		m_vertex_neighbor.resize(6 * vertex_num);
		cuExecute(m_vertex_neighbor.size(),
			AGS_InitVertexNeighbor,
			m_vertex_neighbor);
		cuExecute(m_octree.size(),
			AGS_ComputeVertexNeighbor,
			m_vertex_neighbor,
			m_node2Ver,
			m_octree,
			m_leafIndex,
			m_neighbors);

		printf("AdaptiveGridSet: vertex num %d  \n", vertex_num);
	}

	DEFINE_CLASS(AdaptiveGridSet);
}