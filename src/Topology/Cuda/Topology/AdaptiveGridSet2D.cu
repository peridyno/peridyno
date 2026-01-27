#include "AdaptiveGridSet2D.h"
#include "Algorithm/Reduction.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <thrust/sort.h>

namespace dyno
{
	IMPLEMENT_TCLASS(AdaptiveGridSet2D, TDataType)

	class DeDuplicationHelper
	{
	public:

		DYN_FUNC DeDuplicationHelper()
		{
			surface_index = EMPTY;
			position_index = 0;
		}
		DYN_FUNC DeDuplicationHelper(int surf, OcKey pos)
		{
			surface_index = surf;
			position_index = pos;
		}
		DYN_FUNC bool operator> (const DeDuplicationHelper& ug) const
		{
			return position_index > ug.position_index;
		}
		DYN_FUNC bool isEmpty() { return surface_index == EMPTY; }

		int surface_index;
		OcKey position_index;
	};
	struct PositionCmp
	{
		DYN_FUNC bool operator()(const DeDuplicationHelper& A, const DeDuplicationHelper& B)
		{
			return A > B;
		}
	};

	//the mapping relation: index=(x+1)+(y+1)*3
	__constant__ int offset2D9[9][2] = {
			-1, -1,
			0, -1,
			1, -1,
			-1, 0,
			0, 0,
			1, 0,
			-1, 1,
			0, 1,
			1, 1
	};

	__constant__ int offset2D4[4][2] = {
	-1,0,
	1, 0,
	0,-1,
	0, 1
	};

	template<typename TDataType>
	AdaptiveGridSet2D<TDataType>::AdaptiveGridSet2D()
		: TopologyModule()
	{
	}

	template<typename TDataType>
	AdaptiveGridSet2D<TDataType>::~AdaptiveGridSet2D()
	{
	}

	template<typename TDataType>
	void AdaptiveGridSet2D<TDataType>::clear()
	{
		m_quadtree.clear();
		m_neighbors.clear();
		m_leafIndex.clear();
		//m_vertexs.clear();
		//m_node2ver.clear();
	}

	template<typename TDataType>
	void AdaptiveGridSet2D<TDataType>::setAGrids(DArray<AdaptiveGridNode2D>& nodes)
	{
		m_quadtree.resize(nodes.size());
		m_quadtree.assign(nodes);
	}
	template<typename TDataType>
	void AdaptiveGridSet2D<TDataType>::setNeighbors(DArrayList<int>& nodes)
	{
		m_neighbors.assign(nodes);
	}


	__global__ void AGS2D_CountLeafs(
		DArray<int> leafs,
		DArray<AdaptiveGridNode2D>quadtree)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= leafs.size()) return;

		if (quadtree[tId].isLeaf())
			leafs[tId] = 1;
	}

	template <typename Real,typename Coord2D>
	__global__ void AGS2D_ComputeLeafs(
		DArray<Coord2D> leafs_pos,
		DArray<Real> leafs_scale,
		DArray<int> count,
		DArray<AdaptiveGridNode2D> quadtree,
		Real dx,
		Level lmax)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		if (quadtree[tId].isLeaf())
		{
			leafs_pos[count[tId]] = quadtree[tId].m_position;
			leafs_scale[count[tId]] = dx * (1 << (lmax - quadtree[tId].m_level));
		}
	}

	template<typename TDataType>
	void AdaptiveGridSet2D<TDataType>::extractLeafs(DArray<Coord2D>& pos, DArray<Real>& scale)
	{
		//compute leaf nodes
		pos.resize(m_leaf_num);
		scale.resize(m_leaf_num);
		cuExecute(m_leafIndex.size(),
			AGS2D_ComputeLeafs,
			pos,
			scale,
			m_leafIndex,
			m_quadtree,
			m_dx,
			m_level_max);
	}
	template <typename Coord2D>
	__global__ void AGS2D_ComputePosAndNeighbors(
		DArray<Coord2D> leafs_pos,
		DArray<uint> neighbors_count,
		DArray<int> count,
		DArray<AdaptiveGridNode2D> quadtree,
		DArrayList<int> neighbors)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		if (quadtree[tId].isLeaf())
		{
			leafs_pos[count[tId]] = quadtree[tId].m_position;

			neighbors_count[4 * count[tId]] = neighbors[4 * tId].size();
			neighbors_count[4 * count[tId] + 1] = neighbors[4 * tId + 1].size();
			neighbors_count[4 * count[tId] + 2] = neighbors[4 * tId + 2].size();
			neighbors_count[4 * count[tId] + 3] = neighbors[4 * tId + 3].size();
		}
	}

	__global__ void AGS2D_ComputeLeafsAndNeighbors(
		DArray<AdaptiveGridNode2D> leafs,
		DArray<uint> neighbors_count,
		DArray<int> count,
		DArray<AdaptiveGridNode2D> quadtree,
		DArrayList<int> neighbors)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		if (quadtree[tId].isLeaf())
		{
			leafs[count[tId]] = quadtree[tId];

			neighbors_count[4 * count[tId]] = neighbors[4 * tId].size();
			neighbors_count[4 * count[tId] + 1] = neighbors[4 * tId + 1].size();
			neighbors_count[4 * count[tId] + 2] = neighbors[4 * tId + 2].size();
			neighbors_count[4 * count[tId] + 3] = neighbors[4 * tId + 3].size();
		}
	}

	template <typename Coord2D>
	__global__ void AGS2D_ComputeLeafsAndPosAndNeighbors(
		DArray<Coord2D> leafs_pos,
		DArray<AdaptiveGridNode2D> leafs,
		DArray<uint> neighbors_count,
		DArray<int> count,
		DArray<AdaptiveGridNode2D> quadtree,
		DArrayList<int> neighbors)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		if (quadtree[tId].isLeaf())
		{
			leafs_pos[count[tId]] = quadtree[tId].m_position;
			leafs[count[tId]] = quadtree[tId];

			neighbors_count[4 * count[tId]] = neighbors[4 * tId].size();
			neighbors_count[4 * count[tId] + 1] = neighbors[4 * tId + 1].size();
			neighbors_count[4 * count[tId] + 2] = neighbors[4 * tId + 2].size();
			neighbors_count[4 * count[tId] + 3] = neighbors[4 * tId + 3].size();
		}
	}

	__global__ void AGS2D_ComputeNeighbors(
		DArrayList<int> leafs_neighbors,
		DArray<int> count,
		DArray<AdaptiveGridNode2D> quadtree,
		DArrayList<int> neighbors)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= count.size()) return;

		if (quadtree[tId].isLeaf())
		{
			for (int i = 0; i < neighbors[4 * tId].size(); i++)
				leafs_neighbors[4 * count[tId]].insert(count[(neighbors[4 * tId][i])]);
			for (int i = 0; i < neighbors[4 * tId + 1].size(); i++)
				leafs_neighbors[4 * count[tId] + 1].insert(count[(neighbors[4 * tId + 1][i])]);
			for (int i = 0; i < neighbors[4 * tId + 2].size(); i++)
				leafs_neighbors[4 * count[tId] + 2].insert(count[(neighbors[4 * tId + 2][i])]);
			for (int i = 0; i < neighbors[4 * tId + 3].size(); i++)
				leafs_neighbors[4 * count[tId] + 3].insert(count[(neighbors[4 * tId + 3][i])]);
		}
	}

	template<typename TDataType>
	void AdaptiveGridSet2D<TDataType>::extractLeafs(DArray<Coord2D>& pos, DArrayList<int>& neighbors)
	{
		if (m_neighbors.size() != 4 * m_quadtree.size())
		{
			printf("Please make sure neighbors are constructed? \n");
			return;
		}

		pos.resize(m_leaf_num);
		DArray<uint> count(4 * m_leaf_num);
		cuExecute(m_leafIndex.size(),
			AGS2D_ComputePosAndNeighbors,
			pos,
			count,
			m_leafIndex,
			m_quadtree,
			m_neighbors);

		neighbors.resize(count);
		count.clear();
		cuExecute(m_leafIndex.size(),
			AGS2D_ComputeNeighbors,
			neighbors,
			m_leafIndex,
			m_quadtree,
			m_neighbors);
	}

	template<typename TDataType>
	void AdaptiveGridSet2D<TDataType>::extractLeafs(DArray<AdaptiveGridNode2D>& leaves, DArrayList<int>& neighbors)
	{
		if (m_neighbors.size() != 4 * m_quadtree.size())
		{
			printf("Please make sure neighbors are constructed? \n");
			return;
		}

		leaves.resize(m_leaf_num);
		DArray<uint> count(4 * m_leaf_num);
		cuExecute(m_leafIndex.size(),
			AGS2D_ComputeLeafsAndNeighbors,
			leaves,
			count,
			m_leafIndex,
			m_quadtree,
			m_neighbors);

		neighbors.resize(count);
		count.clear();
		cuExecute(m_leafIndex.size(),
			AGS2D_ComputeNeighbors,
			neighbors,
			m_leafIndex,
			m_quadtree,
			m_neighbors);
	}

	template<typename TDataType>
	void AdaptiveGridSet2D<TDataType>::extractLeafs(DArray<Coord2D>& pos, DArray<AdaptiveGridNode2D>& leaves, DArrayList<int>& neighbors)
	{
		if (m_neighbors.size() != 4 * m_quadtree.size())
		{
			printf("Please make sure neighbors are constructed? \n");
			return;
		}

		pos.resize(m_leaf_num);
		leaves.resize(m_leaf_num);
		DArray<uint> count(4 * m_leaf_num);
		cuExecute(m_leafIndex.size(),
			AGS2D_ComputeLeafsAndPosAndNeighbors,
			pos,
			leaves,
			count,
			m_leafIndex,
			m_quadtree,
			m_neighbors);

		neighbors.resize(count);
		count.clear();
		cuExecute(m_leafIndex.size(),
			AGS2D_ComputeNeighbors,
			neighbors,
			m_leafIndex,
			m_quadtree,
			m_neighbors);
	}

	template <typename TDataType,typename Real,typename Coord2D>
	__global__ void AGS2D_Assess(
		DArray<int> index,
		DArray<Coord2D> pos,
		DArray<AdaptiveGridNode2D> quadtree,
		AdaptiveGridSet2D<TDataType> gridSet,
		DArray<int> leafIndex,
		Coord2D m_origin,
		Real m_dx,
		Level m_levelmax,
		Level m_levelmin,
		int resolution)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= pos.size()) return;

		Coord2D resid = (pos[tId] - m_origin) / m_dx;
		int i = (int)floor(resid[0]);
		int j = (int)floor(resid[1]);
		if ((i < 0 || i >= resolution) || (j < 0 || j >= resolution))
		{
			index[tId] = EMPTY;
			return;
		}

		OcKey morton = CalculateMortonCode2D(i, j);
		int ind = EMPTY;
		gridSet.accessRandom2D(ind, morton, m_levelmax);
		index[tId] = leafIndex[ind];
	}

	template<typename TDataType>
	void AdaptiveGridSet2D<TDataType>::accessRandom(DArray<int>& index, DArray<Coord2D>& pos)
	{
		if (index.size() != pos.size())
			index.resize(pos.size());

		int max_resolution = (1 << m_level_max);
		cuExecute(pos.size(),
			AGS2D_Assess,
			index,
			pos,
			m_quadtree,
			*this,
			m_leafIndex,
			m_origin,
			m_dx,
			m_level_max,
			m_level_max - m_level_num + 1,
			max_resolution);
	}

	__global__ void AGS2D_ExtractLeaves(
		DArray<AdaptiveGridNode2D> leaves,
		DArray<AdaptiveGridNode2D> quadtree,
		DArray<int> count)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		if (quadtree[tId].isLeaf())
		{
			leaves[count[tId]] = quadtree[tId];
		}
	}

	template<typename TDataType>
	void AdaptiveGridSet2D<TDataType>::extractLeafs(DArray<AdaptiveGridNode2D>& leaves)
	{
		//compute leaf nodes
		leaves.resize(m_leaf_num);
		cuExecute(m_leafIndex.size(),
			AGS2D_ExtractLeaves,
			leaves,
			m_quadtree,
			m_leafIndex);
	}

	template<typename TDataType>
	__global__ void AGS2D_CountNeighborhoodForest(
		DArray<uint> ncount,
		DArray<AdaptiveGridNode2D> nodes,
		Level levelmax,
		Level levelmin,
		AdaptiveGridSet2D<TDataType> gridSet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		auto alpha = [&](OcIndex i, OcIndex j, Level gl) -> bool {
			int gresolution = 1 << gl;
			if (i < 0 || i >= gresolution)
				return false;
			if (j < 0 || j >= gresolution)
				return false;

			return true;
			};

		if (nodes[tId].isLeaf())
		{
			Level gl = nodes[tId].m_level;
			OcIndex gnx, gny;
			RecoverFromMortonCode2D((OcKey)nodes[tId].m_morton, gnx, gny);

			for (int c = 0; c < 4; c++)
			{
				if (alpha(gnx + offset2D4[c][0], gny + offset2D4[c][1], gl))
				{
					int nindex = 0;
					OcKey nmorton = CalculateMortonCode2D(gnx + offset2D4[c][0], gny + offset2D4[c][1]);
					if (gridSet.accessRandom2D(nindex,nmorton,gl))
					{

						ncount[tId * 4 + c] = 1;
						if (gl > (nodes[nindex].m_level))
						{
							int nc = c + 1 - 2 * (c % 2);
							ncount[nindex * 4 + nc] = 2;
						}
					}
				}
			}
		}
	}

	template<typename TDataType>
	__global__ void AGS2D_ComputeNeighborhoodForest(
		DArrayList<int> neighbors,
		DArray<AdaptiveGridNode2D> nodes,
		Level levelmax,
		Level levelmin,
		AdaptiveGridSet2D<TDataType> gridSet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		if (nodes[tId].isLeaf())
		{
			Level gl = nodes[tId].m_level;
			OcIndex gnx, gny;
			RecoverFromMortonCode2D((OcKey)nodes[tId].m_morton, gnx, gny);

			int ind = tId - (nodes[tId].m_morton & 3);
			int gresolution = 1 << gl;
			int nindex;
			if (gnx > 0)
			{
				OcKey nmorton = CalculateMortonCode2D(gnx - 1, gny);
				if(gridSet.accessRandom2D(nindex,nmorton,gl))
				{
					neighbors[tId * 4].insert(nindex);

					if (gl > (nodes[nindex].m_level))
					{
						neighbors[nindex * 4 + 1].insert(ind);
						neighbors[nindex * 4 + 1].insert(ind + 2);
					}
				}
			}
			if (gnx < (gresolution - 1))
			{
				OcKey nmorton = CalculateMortonCode2D(gnx + 1, gny);
				if (gridSet.accessRandom2D(nindex, nmorton, gl))
				{
					neighbors[tId * 4 + 1].insert(nindex);

					if (gl > (nodes[nindex].m_level))
					{
						neighbors[nindex * 4].insert(ind + 1);
						neighbors[nindex * 4].insert(ind + 3);
					}
				}
			}
			if (gny > 0)
			{
				OcKey nmorton = CalculateMortonCode2D(gnx, gny - 1);
				if (gridSet.accessRandom2D(nindex, nmorton, gl))
				{
					neighbors[tId * 4 + 2].insert(nindex);

					if (gl > (nodes[nindex].m_level))
					{
						neighbors[nindex * 4 + 3].insert(ind);
						neighbors[nindex * 4 + 3].insert(ind + 1);
					}
				}
			}
			if (gny < (gresolution - 1))
			{
				OcKey nmorton = CalculateMortonCode2D(gnx, gny + 1);
				if (gridSet.accessRandom2D(nindex, nmorton, gl))
				{
					neighbors[tId * 4 + 3].insert(nindex);

					if (gl > (nodes[nindex].m_level))
					{
						neighbors[nindex * 4 + 2].insert(ind + 2);
						neighbors[nindex * 4 + 2].insert(ind + 3);
					}
				}
			}
		}
	}

	template<typename TDataType>
	void AdaptiveGridSet2D<TDataType>::ConstructNeighbors4()
	{
		int node_num = m_quadtree.size();
		DArray<uint> data_count(node_num * 4);
		data_count.reset();

		cuExecute(node_num,
			AGS2D_CountNeighborhoodForest,
			data_count,
			m_quadtree,
			m_level_max,
			m_level_max - m_level_num + 1,
			*this);
		
		uint neighbor_num = thrust::reduce(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), (int)0, thrust::plus<uint>());

		m_neighbors.resize(data_count);
		cuExecute(node_num,
			AGS2D_ComputeNeighborhoodForest,
			m_neighbors,
			m_quadtree,
			m_level_max,
			m_level_max - m_level_num + 1,
			*this);
		data_count.clear();
	}

	template <typename Real,typename Coord2D>
	__global__ void AGS2D_ComputeVertexBuffer(
		DArray<DeDuplicationHelper> vertex,
		DArray<Coord2D> node,
		DArray<Real> scale,
		Coord2D origin,
		Real dx)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= node.size()) return;

		Coord2D vorigin = origin - Coord2D(0.5 * dx, 0.5 * dx);
		Real up_dx = scale[tId];

		auto delta = [&](Coord2D vpos) -> OcKey {
			OcIndex i = (OcIndex)floor((vpos[0] - vorigin[0]) / dx);
			OcIndex j = (OcIndex)floor((vpos[1] - vorigin[1]) / dx);

			return CalculateMortonCode2D(i, j);
			};

		OcKey v1 = delta(node[tId] + Coord2D(-0.5 * up_dx, -0.5 * up_dx));
		vertex[4 * tId] = DeDuplicationHelper(tId, v1);

		OcKey v2 = delta(node[tId] + Coord2D(0.5 * up_dx, -0.5 * up_dx));
		vertex[4 * tId + 1] = DeDuplicationHelper(tId, v2);

		OcKey v3 = delta(node[tId] + Coord2D(0.5 * up_dx, 0.5 * up_dx));
		vertex[4 * tId + 2] = DeDuplicationHelper(tId, v3);

		OcKey v4 = delta(node[tId] + Coord2D(-0.5 * up_dx, 0.5 * up_dx));
		vertex[4 * tId + 3] = DeDuplicationHelper(tId, v4);
	}

	__global__ void AGS2D_CountNonRepeatedVertex(
		DArray<uint> count,
		DArray<DeDuplicationHelper> vertex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertex.size()) return;

		if ((tId == 0 || vertex[tId].position_index != vertex[tId - 1].position_index))
			count[tId] = 1;
	}

	template <typename Real,typename Coord2D>
	__global__ void AGS2D_ComputeNodeToVertex(
		DArray<int> Node2Ver,
		DArray<uint> vcount,
		DArray<Coord2D> vpos,
		DArray<uint> count,
		DArray<DeDuplicationHelper> vertex,
		DArray<Coord2D> node,
		Coord2D origin,
		Real dx)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertex.size()) return;

		OcIndex nx, ny;
		RecoverFromMortonCode2D(OcKey(vertex[tId].position_index), nx, ny);
		Coord2D vertexpos(origin[0] + nx * dx, origin[1] + ny * dx);
		int vindex;
		if ((tId == 0 || vertex[tId].position_index != vertex[tId - 1].position_index))
		{
			vindex = count[tId];
			vpos[vindex] = vertexpos;
		}
		else
			vindex = count[tId] - 1;

		int nindex = vertex[tId].surface_index;
		Coord2D nodepos = node[nindex];
		if (vertexpos[0] < nodepos[0] && vertexpos[1] < nodepos[1])
			Node2Ver[4 * nindex] = vindex;
		else if (vertexpos[0] > nodepos[0] && vertexpos[1] < nodepos[1])
			Node2Ver[4 * nindex + 1] = vindex;
		else if (vertexpos[0] > nodepos[0] && vertexpos[1] > nodepos[1])
			Node2Ver[4 * nindex + 2] = vindex;
		else if (vertexpos[0] < nodepos[0] && vertexpos[1] > nodepos[1])
			Node2Ver[4 * nindex + 3] = vindex;

		atomicAdd(&(vcount[vindex]), 1);
	}

	__global__ void AGS2D_ComputeVertexToNode(
		DArrayList<int> Ver2Node,
		DArray<uint> count,
		DArray<DeDuplicationHelper> vertex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertex.size()) return;

		int vindex;
		if ((tId == 0 || vertex[tId].position_index != vertex[tId - 1].position_index))
			vindex = count[tId];
		else
			vindex = count[tId] - 1;

		int nindex = vertex[tId].surface_index;

		Ver2Node[vindex].atomicInsert(nindex);
	}

	template<typename TDataType>
	void AdaptiveGridSet2D<TDataType>::extractVertexs(DArray<Coord2D>& vertex, DArray<int>& n2v, DArrayList<int>& ver2node)
	{
		DArray<Coord2D> leaf_node;
		DArray<Real> leaf_scale;
		extractLeafs(leaf_node, leaf_scale);

		int node_num = leaf_node.size();
		DArray<DeDuplicationHelper> vertex_buf(4 * node_num);
		cuExecute(node_num,
			AGS2D_ComputeVertexBuffer,
			vertex_buf,
			leaf_node,
			leaf_scale,
			m_origin,
			m_dx);
		thrust::sort(thrust::device, vertex_buf.begin(), vertex_buf.begin() + vertex_buf.size(), PositionCmp());

		DArray<uint> count(vertex_buf.size());
		count.reset();
		cuExecute(count.size(),
			AGS2D_CountNonRepeatedVertex,
			count,
			vertex_buf);
		Reduction<uint> reduce;
		int vertex_num = reduce.accumulate(count.begin(), count.size());
		Scan<uint> scan;
		scan.exclusive(count.begin(), count.size());

		vertex.resize(vertex_num);
		n2v.resize(4 * node_num);
		DArray<uint> vertex_count(vertex_num);
		vertex_count.reset();
		cuExecute(count.size(),
			AGS2D_ComputeNodeToVertex,
			n2v,
			vertex_count,
			vertex,
			count,
			vertex_buf,
			leaf_node,
			m_origin,
			m_dx);

		ver2node.resize(vertex_count);
		cuExecute(count.size(),
			AGS2D_ComputeVertexToNode,
			ver2node,
			count,
			vertex_buf);

		leaf_node.clear();
		leaf_scale.clear();
		vertex_buf.clear();
		count.clear();
		vertex_count.clear();
	}

	template<typename TDataType>
	void AdaptiveGridSet2D<TDataType>::updateTopology()
	{
		m_leafIndex.resize(m_quadtree.size());
		m_leafIndex.reset();
		//count the number of leaf nodes
		cuExecute(m_leafIndex.size(),
			AGS2D_CountLeafs,
			m_leafIndex,
			m_quadtree);

		m_leaf_num = thrust::reduce(thrust::device, m_leafIndex.begin(), m_leafIndex.begin() + m_leafIndex.size(), (int)0, thrust::plus<uint>());
		thrust::exclusive_scan(thrust::device, m_leafIndex.begin(), m_leafIndex.begin() + m_leafIndex.size(), m_leafIndex.begin());
		printf("AdaptiveGridSet: leaf num %d %d \n", m_leaf_num, m_quadtree.size());

		ConstructNeighbors4();
	}

	DEFINE_CLASS(AdaptiveGridSet2D);
}