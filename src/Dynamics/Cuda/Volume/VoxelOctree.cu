#include "VoxelOctree.h"
#include "Algorithm/Reduction.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <thrust/sort.h>

namespace dyno
{
	IMPLEMENT_TCLASS(VoxelOctree, TDataType)

	template<typename TDataType>
	VoxelOctree<TDataType>::VoxelOctree()
		: TopologyModule()
	{
	}

	template<typename TDataType>
	VoxelOctree<TDataType>::~VoxelOctree()
	{
	}

	template<typename TDataType>
	void VoxelOctree<TDataType>::setVoxelOctree(DArray<VoxelOctreeNode<Coord>>& oct)
	{
		m_octree.resize(oct.size());
		m_octree.assign(oct);

		tagAsChanged();
	}

	template <typename Coord>
	__global__ void SO_CountLeafs(
		DArray<int> leafs,
		DArray<VoxelOctreeNode<Coord>> octree)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= leafs.size()) return;

		if (!octree[tId].midside())
			leafs[tId] = 1;
	}

	template <typename Coord>
	__global__ void SO_ComputeLeafs(
		DArray<Coord> leafs_pos,
		DArray<int> leafs_sdf,
		DArray<int> leafs,
		DArray<VoxelOctreeNode<Coord>> octree)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= leafs.size()) return;

		if (!octree[tId].midside())
		{
			leafs_pos[leafs[tId]] = octree[tId].position();
			//leafs_sdf[leafs[tId]] = octree[tId].value();
			leafs_sdf[leafs[tId]] = tId;
		}
		//printf("voxelOctree: tId pos %d %f %f %f \n",tId, leafs_pos[leafs[tId]][0], leafs_pos[leafs[tId]][1], leafs_pos[leafs[tId]][2]);
	}

	template<typename TDataType>
	void VoxelOctree<TDataType>::getLeafs(DArray<Coord>& pos, DArray<int>& pos_pos)
	{
		DArray<int> points_count;
		points_count.resize(m_octree.size());
		points_count.reset();
		//数叶子节点的个数
		cuExecute(points_count.size(),
			SO_CountLeafs,
			points_count,
			m_octree);

		int leafs_num = thrust::reduce(thrust::device, points_count.begin(), points_count.begin() + points_count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, points_count.begin(), points_count.begin() + points_count.size(), points_count.begin());
		std::printf("GetLeafs: the number of leafs is: %d \n", leafs_num);

		DArray<Coord> points_pos;
		points_pos.resize(leafs_num);
		DArray<int> points_sdf;
		points_sdf.resize(leafs_num);
		//取叶节点的坐标
		cuExecute(points_count.size(),
			SO_ComputeLeafs,
			points_pos,
			points_sdf,
			points_count,
			m_octree);

		pos.assign(points_pos);
		pos_pos.assign(points_sdf);

		points_count.clear();
		points_pos.clear();
		points_sdf.clear();
	}

	template <typename Real, typename Coord>
	__global__ void VO_ComputeVertices(
		DArray<Coord> vertices,
		DArray<Coord> centers,
		DArray<int> leaves,
		DArray<VoxelOctreeNode<Coord>> octree,
		Real dx)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= centers.size()) return;
		Real coef0 = Real(pow(Real(2), int(octree[leaves[tId]].level())));
		Real h = coef0 * dx;
		
		Coord p = centers[tId] - 0.5 * h;
		vertices[8*tId] = p;
		vertices[8*tId + 1] = p + Coord(h, 0, 0);
		vertices[8*tId + 2] = p + Coord(h, h, 0);
		vertices[8*tId + 3] = p + Coord(0, h, 0);
		vertices[8*tId + 4] = p + Coord(0, 0, h);
		vertices[8*tId + 5] = p + Coord(h, 0, h);
		vertices[8*tId + 6] = p + Coord(h, h, h);
		vertices[8*tId + 7] = p + Coord(0, h, h);
	}

	template<typename TDataType>
	void VoxelOctree<TDataType>::getCellVertices(DArray<Coord>& pos)
	{
		DArray<Coord> centers;
		DArray<int> leaves;
		this->getLeafs(centers, leaves);
		uint cellSize = centers.size();
		pos.resize(8 * cellSize);
		
		cuExecute(cellSize,
			VO_ComputeVertices,
			pos,
			centers,
			leaves,
			m_octree,
			m_dx);
		centers.clear();
		leaves.clear();
	}

	template <typename Real, typename Coord>
	__global__ void VO_ComputeVertices0(
		DArray<Coord> vertices,
		DArray<VoxelOctreeNode<Coord>> octree,
		Real dx,
		int bottom_size)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= bottom_size) return;

		Coord p = octree[tId].position() - 0.5 * dx;

		vertices[8 * tId] = p;
		vertices[8 * tId + 1] = p + Coord(dx, 0, 0);
		vertices[8 * tId + 2] = p + Coord(dx, dx, 0);
		vertices[8 * tId + 3] = p + Coord(0, dx, 0);
		vertices[8 * tId + 4] = p + Coord(0, 0, dx);
		vertices[8 * tId + 5] = p + Coord(dx, 0, dx);
		vertices[8 * tId + 6] = p + Coord(dx, dx, dx);
		vertices[8 * tId + 7] = p + Coord(0, dx, dx);
	}

	template<typename TDataType>
	void VoxelOctree<TDataType>::getCellVertices0(DArray<Coord>& pos)
	{
		pos.resize(8 * m_level0);

		cuExecute(m_level0,
			VO_ComputeVertices0,
			pos,
			m_octree,
			m_dx,
			m_level0);
	}

	template <typename Real, typename Coord>
	__global__ void VO_ComputeVertices1(
		DArray<Coord> vertices,
		int nx,
		int ny,
		int nz,
		Real dx,
		Coord origin)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= vertices.size()) return;

		int i = tId % nx;
		int j = ((tId - i) / nx) % ny;
		int k = (tId - i - j * nx) / (nx*ny);

		Coord p = origin + Coord(i*dx, j*dx, k*dx);

		vertices[8 * tId] = p;
		vertices[8 * tId + 1] = p + Coord(dx, 0, 0);
		vertices[8 * tId + 2] = p + Coord(dx, dx, 0);
		vertices[8 * tId + 3] = p + Coord(0, dx, 0);
		vertices[8 * tId + 4] = p + Coord(0, 0, dx);
		vertices[8 * tId + 5] = p + Coord(dx, 0, dx);
		vertices[8 * tId + 6] = p + Coord(dx, dx, dx);
		vertices[8 * tId + 7] = p + Coord(0, dx, dx);

		//printf("point: %d %d %d, %d, %d %d %d, %f %f %f \n", nx, ny, nz, tId, i, j, k, p[0], p[1], p[2]);
	}

	template<typename TDataType>
	void VoxelOctree<TDataType>::getCellVertices1(DArray<Coord>& pos)
	{
		int num = m_nx * m_ny*m_nz;
		pos.resize(8 * num);

		cuExecute(num,
			VO_ComputeVertices1,
			pos,
			m_nx,
			m_ny,
			m_nz,
			m_dx,
			m_origin);
	}

	template <typename Coord>
	__global__ void VO_CountVertices2(
		DArray<int> count,
		DArray<VoxelOctreeNode<Coord>> octree,
		int bottom_size)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= bottom_size) return;

		int x1y0z0 = octree[tId].m_neighbor[1];
		if (x1y0z0 == EMPTY) return;

		int x0y1z0 = octree[tId].m_neighbor[3];
		if (x0y1z0 == EMPTY) return;

		int x0y0z1 = octree[tId].m_neighbor[5];
		if (x0y0z1 == EMPTY) return;

		int x1y1z0=octree[x1y0z0].m_neighbor[3];
		if (x1y1z0 == EMPTY) return;

		int x1y0z1 = octree[x1y0z0].m_neighbor[5];
		if (x1y0z1 == EMPTY) return;

		int x1y1z1 = octree[x1y1z0].m_neighbor[5];
		if (x1y1z1 == EMPTY) return;

		int x0y1z1 = octree[x0y0z1].m_neighbor[3];
		if (x0y1z1 == EMPTY) return;

		count[tId] = 1;
	}

	template <typename Coord>
	__global__ void VO_ComputeVertices2(
		DArray<Coord> vertices,
		DArray<VoxelOctreeNode<Coord>> octree,
		DArray<int> count)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= count.size()) return;

		if (count[tId] == 0 || count[tId] != count[tId - 1])
		{
			int x1y0z0 = octree[tId].m_neighbor[1];
			int x0y1z0 = octree[tId].m_neighbor[3];
			int x0y0z1 = octree[tId].m_neighbor[5];
			int x1y1z0 = octree[x1y0z0].m_neighbor[3];
			int x1y0z1 = octree[x1y0z0].m_neighbor[5];
			int x1y1z1 = octree[x1y1z0].m_neighbor[5];
			int x0y1z1 = octree[x0y0z1].m_neighbor[3];

			vertices[8 * count[tId]] = octree[tId].position();
			vertices[8 * count[tId] + 1] = octree[x1y0z0].position();
			vertices[8 * count[tId] + 2] = octree[x1y1z0].position();
			vertices[8 * count[tId] + 3] = octree[x0y1z0].position();
			vertices[8 * count[tId] + 4] = octree[x0y0z1].position();
			vertices[8 * count[tId] + 5] = octree[x1y0z1].position();
			vertices[8 * count[tId] + 6] = octree[x1y1z1].position();
			vertices[8 * count[tId] + 7] = octree[x0y1z1].position();
		}
	}

	template<typename TDataType>
	void VoxelOctree<TDataType>::getCellVertices2(DArray<Coord>& pos)
	{
		DArray<int> count;
		count.resize(m_level0);
		cuExecute(m_level0,
			VO_CountVertices2,
			count,
			m_octree,
			m_level0);
		int grid_num = thrust::reduce(thrust::device, count.begin(), count.begin() + count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, count.begin(), count.begin() + count.size(), count.begin());

		printf("the grid number is:  %d \n", grid_num);

		pos.resize(8 * grid_num);
		cuExecute(m_level0,
			VO_ComputeVertices2,
			pos,
			m_octree,
			count);
	}

	//axis_index: 0(x-1),1(x+1),2(y-1),3(y+1),4(z-1),5(z+1)
	template <typename Coord>
	GPU_FUNC void VO_NeighborIteration(
		DArray<uint>& count,
		DArray<VoxelOctreeNode<Coord>>& octree,
		int& num,
		int index,
		int axis_index)
	{
		if (octree[index].midside() == true)
		{
			num += 4;
			int child_index = octree[index].child();
			if (axis_index == 0)
			{
				VO_NeighborIteration(count, octree, num, child_index + 6, 0);
				VO_NeighborIteration(count, octree, num, child_index + 4, 0);
				VO_NeighborIteration(count, octree, num, child_index + 2, 0);
				VO_NeighborIteration(count, octree, num, child_index + 0, 0);
			}
			else if (axis_index == 1)
			{
				VO_NeighborIteration(count, octree, num, child_index + 7, 1);
				VO_NeighborIteration(count, octree, num, child_index + 5, 1);
				VO_NeighborIteration(count, octree, num, child_index + 3, 1);
				VO_NeighborIteration(count, octree, num, child_index + 1, 1);
			}
			else if (axis_index == 2)
			{
				VO_NeighborIteration(count, octree, num, child_index + 5, 2);
				VO_NeighborIteration(count, octree, num, child_index + 4, 2);
				VO_NeighborIteration(count, octree, num, child_index + 1, 2);
				VO_NeighborIteration(count, octree, num, child_index + 0, 2);
			}
			else if (axis_index == 3)
			{
				VO_NeighborIteration(count, octree, num, child_index + 7, 3);
				VO_NeighborIteration(count, octree, num, child_index + 6, 3);
				VO_NeighborIteration(count, octree, num, child_index + 3, 3);
				VO_NeighborIteration(count, octree, num, child_index + 2, 3);
			}
			else if (axis_index == 4)
			{
				VO_NeighborIteration(count, octree, num, child_index + 3, 4);
				VO_NeighborIteration(count, octree, num, child_index + 2, 4);
				VO_NeighborIteration(count, octree, num, child_index + 1, 4);
				VO_NeighborIteration(count, octree, num, child_index + 0, 4);
			}
			else if (axis_index == 5)
			{
				VO_NeighborIteration(count, octree, num, child_index + 7, 5);
				VO_NeighborIteration(count, octree, num, child_index + 6, 5);
				VO_NeighborIteration(count, octree, num, child_index + 5, 5);
				VO_NeighborIteration(count, octree, num, child_index + 4, 5);
			}
		}
		else
		{
			atomicAdd(&(count[index]), 1);
		}
	}

	template <typename Coord>
	__global__ void VO_CountNeighbors(
		DArray<uint> count,
		DArray<int> leafs,
		DArray<VoxelOctreeNode<Coord>> octree)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= leafs.size()) return;

		VoxelOctreeNode<Coord> node0 = octree[leafs[tId]];
		int neighbor_num = 0;

		for (int i = 0; i < 6; i++)
		{
			if (node0.m_neighbor[i] != EMPTY)
				VO_NeighborIteration(count, octree, neighbor_num, node0.m_neighbor[i], i);
		}

		atomicAdd(&(count[leafs[tId]]), neighbor_num);
	}

	//axis_index: 0(x-1),1(x+1),2(y-1),3(y+1),4(z-1),5(z+1)
	template <typename Coord>
	GPU_FUNC void VO_GetNeighborIteration(
		DArrayList<int>& neighbors,
		DArray<VoxelOctreeNode<Coord>>& octree,
		int index0,
		int index,
		int axis_index,
		bool is_same_level)
	{
		if (octree[index].midside() == true)
		{
			int child_index = octree[index].child();
			if (axis_index == 0)
			{
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 6, 0, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 4, 0, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 2, 0, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 0, 0, false);
			}
			else if (axis_index == 1)
			{
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 7, 1, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 5, 1, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 3, 1, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 1, 1, false);
			}
			else if (axis_index == 2)
			{
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 5, 2, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 4, 2, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 1, 2, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 0, 2, false);
			}
			else if (axis_index == 3)
			{
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 7, 3, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 6, 3, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 3, 3, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 2, 3, false);
			}
			else if (axis_index == 4)
			{
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 3, 4, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 2, 4, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 1, 4, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 0, 4, false);
			}
			else if (axis_index == 5)
			{
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 7, 5, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 6, 5, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 5, 5, false);
				VO_GetNeighborIteration(neighbors, octree, index0, child_index + 4, 5, false);
			}
		}
		else
		{
			if (is_same_level == false)
			{
				List<int>& list1 = neighbors[index0];
				list1.atomicInsert(index);
			}

			List<int>& list2 = neighbors[index];
			list2.atomicInsert(index0);
		}
	}

	template <typename Coord>
	__global__ void VO_GetNeighbors(
		DArrayList<int> neighbors,
		DArray<int> leafs,
		DArray<VoxelOctreeNode<Coord>> octree)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= leafs.size()) return;

		VoxelOctreeNode<Coord> node0 = octree[leafs[tId]];

		for (int i = 0; i < 6; i++)
		{
			if (node0.m_neighbor[i] != EMPTY)
				VO_GetNeighborIteration(neighbors, octree, leafs[tId], node0.m_neighbor[i], i, true);
		}
	}

	template<typename TDataType>
	void VoxelOctree<TDataType>::updateNeighbors()
	{
		//printf("Update neighbors!!!~~~ \n");

		DArray<Coord> leafs;
		DArray<int> leafs_index;

		this->getLeafs(leafs, leafs_index);

		DArray<uint> count;
		count.resize(m_octree.size());
		count.reset();

		cuExecute(leafs.size(),
			VO_CountNeighbors,
			count,
			leafs_index,
			m_octree);

		m_neighbors.resize(count);
		cuExecute(leafs.size(),
			VO_GetNeighbors,
			m_neighbors,
			leafs_index,
			m_octree);

		leafs.clear();
		leafs_index.clear();
		count.clear();
	}

	template <typename Real>
	__global__ void SO_GetLeafsValue(
		DArray<Real> leafs_sdf,
		DArray<int> leafs_count,
		DArray<Real> octree_value)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= leafs_count.size()) return;

		leafs_sdf[tId] = octree_value[leafs_count[tId]];
	}

	template<typename TDataType>
	void VoxelOctree<TDataType>::setSdfValues(DArray<Real>& vals)
	{
		sdfValues.assign(vals);
	}

	template<typename TDataType>
	void VoxelOctree<TDataType>::getLeafsValue(DArray<Coord>& pos, DArray<Real>& val)
	{
		DArray<int> leafs_count;
		this->getLeafs(pos, leafs_count);

		val.resize(leafs_count.size());


		cuExecute(leafs_count.size(),
			SO_GetLeafsValue,
			val,
			leafs_count,
			sdfValues);

		leafs_count.clear();
	}

	//template <typename Coord>
	//__global__ void SO_BWInitial(
	//	DArray<int> bw,
	//	DArray<int> bw_count,
	//	DArray<VoxelOctreeNode<Coord>> octree)
	//{
	//	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	//	if (tId >= bw.size()) return;
	//	if (tId == 0 || (octree[tId].level() != octree[tId - 1].level()))
	//	{
	//		bw[tId] = 1;
	//		bw_count[tId] = 1;
	//	}
	//}

	//template <typename Coord>
	//__global__ void SO_BWIteration(
	//	DArray<int> bw,
	//	DArray<int> bw_count,
	//	DArrayList<int> neighbor,
	//	DArray<VoxelOctreeNode<Coord>> octree)
	//{
	//	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	//	if (tId >= bw.size()) return;
	//	if (bw_count[tId] == 1)
	//	{
	//		for (int i = 0; i < 6; i++)
	//		{
	//			//printf("iteration: %d %d %d,%d %d \n", tId, i, neighbor.size(), neighbor[i], bw_count[neighbor[i]]);
	//			int id = octree[tId].m_neighbor[i];
	//			if (id > 0 && bw_count[id] == 0)
	//			{
	//				if (bw[tId] == 0)
	//					bw[id] = 1;
	//				bw_count[id] = 1;
	//			}
	//		}
	//	}
	//}

	//template<typename TDataType>
	//void VoxelOctree<TDataType>::getOctreeBW(DArray<int>& nodes_bw)
	//{
	//	DArrayList<int>& leafs_neighbors = this->getNeighbors();
	//	uint num = size();
	//	DArray<int> nodes_count;
	//	nodes_bw.resize(num);
	//	nodes_count.resize(num);
	//	nodes_bw.reset();
	//	nodes_count.reset();
	//	int tnum = 0;
	//	Reduction<int> reduce;
	//	cuExecute(num,
	//		SO_BWInitial,
	//		nodes_bw,
	//		nodes_count,
	//		this->getVoxelOctree());
	//	tnum = reduce.accumulate(nodes_count.begin(), nodes_count.size());
	//	while (tnum < num)
	//	{
	//		cuExecute(num,
	//			SO_BWIteration,
	//			nodes_bw,
	//			nodes_count,
	//			leafs_neighbors,
	//			this->getVoxelOctree());
	//		tnum = reduce.accumulate(nodes_count.begin(), nodes_count.size());
	//	}
	//	nodes_count.clear();
	//}


	template <typename Real>
	GPU_FUNC Real lerp(Real p, Real p0, Real p1, Real v0, Real v1)
	{
		if ((p1 - p0) < REAL_EPSILON)
			return v0;
		else
			return ((p1 - p) * v0 + (p - p0) * v1) / (p1 - p0);
	}

	template <typename Real, typename Coord, typename TDataType>
	__global__ void SO_GetSignDistance(
		DArray<Real> point_sdf,
		DArray<Coord> point_normal,
		DArray<Coord> point_pos,
		VoxelOctree<TDataType> oct_topology,
		DArray<Real> oct_value,
		Coord origin,
		Real dx,
		int nx,
		int ny,
		int nz,
		bool inverted)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= point_pos.size()) return;

		Coord po = (point_pos[tId] - origin) / dx;

		int i = (int)std::floor(po[0]);
		int j = (int)std::floor(po[1]);
		int k = (int)std::floor(po[2]);

		//TODO: check the correctness
		if (i < 0 || i >= nx - 1 || j < 0 || j >= ny - 1 || k < 0 || k >= nz - 1)
		{
			if (inverted == true)
				point_sdf[tId] = -100000.0f;
			else
				point_sdf[tId] = 100000.0f;
			point_normal[tId] = Coord(0);
			return;
		}
		int node_id[8];
		VoxelOctreeNode<Coord> node[8];
		oct_topology.getNode(i, j, k, node[0], node_id[0]);
		oct_topology.getNode(i + 1, j, k, node[1], node_id[1]);
		oct_topology.getNode(i, j + 1, k, node[2], node_id[2]);
		oct_topology.getNode(i + 1, j + 1, k, node[3], node_id[3]);
		oct_topology.getNode(i, j, k + 1, node[4], node_id[4]);
		oct_topology.getNode(i + 1, j, k + 1, node[5], node_id[5]);
		oct_topology.getNode(i, j + 1, k + 1, node[6], node_id[6]);
		oct_topology.getNode(i + 1, j + 1, k + 1, node[7], node_id[7]);

		Real dx00 = lerp(point_pos[tId][0], node[0].position()[0], node[1].position()[0], oct_value[node[0].value()], oct_value[node[1].value()]);
		Real dx10 = lerp(point_pos[tId][0], node[2].position()[0], node[3].position()[0], oct_value[node[2].value()], oct_value[node[3].value()]);
		Real dxy0 = lerp(point_pos[tId][1], node[0].position()[1], node[2].position()[1], dx00, dx10);

		Real dx01 = lerp(point_pos[tId][0], node[4].position()[0], node[5].position()[0], oct_value[node[4].value()], oct_value[node[5].value()]);
		Real dx11 = lerp(point_pos[tId][0], node[6].position()[0], node[7].position()[0], oct_value[node[6].value()], oct_value[node[7].value()]);
		Real dxy1 = lerp(point_pos[tId][1], node[0].position()[1], node[2].position()[1], dx01, dx11);

		Real d0y0 = lerp(point_pos[tId][1], node[0].position()[1], node[2].position()[1], oct_value[node[0].value()], oct_value[node[2].value()]);
		Real d0y1 = lerp(point_pos[tId][1], node[4].position()[1], node[6].position()[1], oct_value[node[4].value()], oct_value[node[6].value()]);
		Real d0yz = lerp(point_pos[tId][2], node[0].position()[2], node[4].position()[2], d0y0, d0y1);

		Real d1y0 = lerp(point_pos[tId][1], node[1].position()[1], node[3].position()[1], oct_value[node[1].value()], oct_value[node[3].value()]);
		Real d1y1 = lerp(point_pos[tId][1], node[5].position()[1], node[7].position()[1], oct_value[node[5].value()], oct_value[node[7].value()]);
		Real d1yz = lerp(point_pos[tId][2], node[0].position()[2], node[4].position()[2], d1y0, d1y1);

		Real dx0z = lerp(point_pos[tId][2], node[0].position()[2], node[4].position()[2], dx00, dx01);
		Real dx1z = lerp(point_pos[tId][2], node[0].position()[2], node[4].position()[2], dx10, dx11);

		point_normal[tId][0] = d0yz - d1yz;
		point_normal[tId][1] = dx0z - dx1z;
		point_normal[tId][2] = dxy0 - dxy1;

		Real l = point_normal[tId].norm();
		if (l < 0.0001f) point_normal[tId] = Coord(0);
		else point_normal[tId] = point_normal[tId].normalize();

		if (inverted == true)
		{
			point_sdf[tId] = -lerp(point_pos[tId][2], node[0].position()[2], node[4].position()[2], dxy0, dxy1);
			point_normal[tId] = -point_normal[tId];
		}
		else
			point_sdf[tId] = lerp(point_pos[tId][2], node[0].position()[2], node[4].position()[2], dxy0, dxy1);
	}


	template<typename TDataType>
	void VoxelOctree<TDataType>::getSignDistance(
		DArray<Coord> point_pos,
		DArray<Real>& point_sdf,
		DArray<Coord>& point_normal,
		bool inverted)
	{
		point_sdf.resize(point_pos.size());
		point_normal.resize(point_pos.size());

		int nx, ny, nz;
		//auto oct = this->stateSDFTopology()->getDataPtr();
		getGrid(nx, ny, nz);

		cuExecute(point_pos.size(),
			SO_GetSignDistance,
			point_sdf,
			point_normal,
			point_pos,
			*this,
			this->getSdfValues(),
			this->getOrigin(),
			this->getDx(),
			nx,
			ny,
			nz,
			inverted);
	}

	DYN_FUNC static void kernel1(Real& val, Real val_x)
	{
		if (std::abs(val_x) < 1)
			val = (1 - std::abs(val_x));
		else
			val = 0;
	}
	DYN_FUNC static void kernel2(Real& val, Real val_x)
	{
		if (std::abs(val_x) < 0.5)
			val = (0.75 - (std::abs(val_x) * std::abs(val_x)));
		else if (std::abs(val_x) < 1.5)
			val = 0.5 * (1.5 - (std::abs(val_x))) * (1.5 - (std::abs(val_x)));
		else
			val = 0;
	}
	DYN_FUNC static void kernel3(Real& val, Real val_x)
	{
		if (std::abs(val_x) < 1)
			val = ((0.5 * abs(val_x) * abs(val_x) * abs(val_x)) - (abs(val_x) * abs(val_x)) + (2.0f / 3.0f));
		else if (std::abs(val_x) < 2)
			val = (1.0f / 6.0f) * (2.0f - (std::abs(val_x))) * (2.0f - (std::abs(val_x))) * (2.0f - (std::abs(val_x)));
		else
			val = 0.0f;
	}


	template <typename Real, typename Coord>
	DYN_FUNC void SO_InterpolationFunction(
		Real& weight,
		Coord point_pos,
		Coord grid_pos,
		Real grid_spacing)
	{
		Real w1, w2, w3;
		kernel2(w1, (point_pos[0] - grid_pos[0]) / grid_spacing);
		kernel2(w2, (point_pos[1] - grid_pos[1]) / grid_spacing);
		kernel2(w3, (point_pos[2] - grid_pos[2]) / grid_spacing);

		weight = w1 * w2 * w3;
	}

	template <typename Real, typename Coord, typename TDataType>
	__global__ void SO_GetSignDistanceKernel(
		DArray<Real> point_sdf,
		DArray<Coord> point_pos,
		VoxelOctree<TDataType> octree_node,
		DArray<Real> octree_value,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= point_pos.size()) return;
		Coord poi = point_pos[tId] - origin_;
		int nx_pos = clamp(int(std::floor(poi[0] / dx_)), 0, nx_ - 1);
		int ny_pos = clamp(int(std::floor(poi[1] / dx_)), 0, ny_ - 1);
		int nz_pos = clamp(int(std::floor(poi[2] / dx_)), 0, nz_ - 1);
		Real coef = Real(pow(Real(2), int(octree_node.getLevelNum())));
		Real dx_top = dx_ * coef;
		int node0_id;
		VoxelOctreeNode<Coord> node0;
		octree_node.getNode(nx_pos, ny_pos, nz_pos, node0, node0_id);
		if ((node0.position() - point_pos[tId]).norm() < REAL_EPSILON)
		{
			point_sdf[tId] = octree_value[node0.value()];
			return;
		}
		bool mask[6] = { 0,0,0,0,0,0 };
		int node_id[6];
		VoxelOctreeNode<Coord> node[6];
		if (nx_pos > 0)
		{
			octree_node.getNode(nx_pos - 1, ny_pos, nz_pos, node[0], node_id[0]);
			mask[0] = 1;
		}
		if (nx_pos < nx_ - 1)
		{
			octree_node.getNode(nx_pos + 1, ny_pos, nz_pos, node[1], node_id[1]);
			mask[1] = 1;
		}
		if (ny_pos > 0)
		{
			octree_node.getNode(nx_pos, ny_pos - 1, nz_pos, node[2], node_id[2]);
			mask[2] = 1;
		}
		if (ny_pos < ny_ - 1)
		{
			octree_node.getNode(nx_pos, ny_pos + 1, nz_pos, node[3], node_id[3]);
			mask[3] = 1;
		}
		if (nz_pos > 0)
		{
			octree_node.getNode(nx_pos, ny_pos, nz_pos - 1, node[4], node_id[4]);
			mask[4] = 1;
		}
		if (nz_pos < nz_ - 1)
		{
			octree_node.getNode(nx_pos, ny_pos, nz_pos + 1, node[5], node_id[5]);
			mask[5] = 1;
		}
		Real weight[7] = { 0,0,0,0,0,0,0 };
		SO_InterpolationFunction(weight[0], point_pos[tId], node0.position(), dx_top);
		for (int i = 1; i < 7; i++)
		{
			if (mask[i] == true)
				SO_InterpolationFunction(weight[i], point_pos[tId], node[i - 1].position(), dx_top);
		}
		Real weight_sum = weight[0] + weight[1] + weight[2] + weight[3] + weight[4] + weight[5] + weight[6];
		//printf("the weight:%d %f %f %f %f %f %f %f %f \n", tId, weight[0], weight[1], weight[2], weight[3], weight[4], weight[5], weight[6], weight_sum);
		if (weight_sum < REAL_EPSILON)
		{
			point_sdf[tId] = octree_value[node0.value()];
			return;
		}
		else
		{
			Real sdf_value = octree_value[node0.value()] * weight[0] / weight_sum;;
			for (int i = 1; i < 7; i++)
			{
				if (mask[i - 1] == true)
					sdf_value += octree_value[node[i - 1].value()] * weight[i] / weight_sum;
			}
			point_sdf[tId] = sdf_value;
			return;
		}
	}

	template<typename TDataType>
	void VoxelOctree<TDataType>::getSignDistanceKernel(
		DArray<Coord> point_pos,
		DArray<Real>& point_sdf)
	{
		point_sdf.resize(point_pos.size());

		int nx, ny, nz;
		this->getGrid(nx, ny, nz);

		cuExecute(point_pos.size(),
			SO_GetSignDistanceKernel,
			point_sdf,
			point_pos,
			*this,
			this->getSdfValues(),
			this->getOrigin(),
			this->getDx(),
			nx,
			ny,
			nz);
	}

	template <typename Real, typename Coord, typename TDataType>
	__global__ void SO_GetSignDistanceMLS(
		DArray<Real> point_sdf,
		DArray<Coord> point_normal,
		DArray<Coord> point_pos,
		VoxelOctree<TDataType> octree_node,
		DArrayList<int> octree_neighbors,
		DArray<Real> octree_value,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_,
		bool inverted)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= point_pos.size()) return;

		Coord pos = point_pos[tId];
		Coord po = (pos - origin_) / dx_;

		int pi = (int)std::floor(po[0]);
		int pj = (int)std::floor(po[1]);
		int pk = (int)std::floor(po[2]);

		//TODO: check the correctness
		if (pi < 0 || pi > nx_ - 1 || pj < 0 || pj > ny_ - 1 || pk < 0 || pk > nz_ - 1)
		{
			if (inverted == true)
				point_sdf[tId] = -100000.0f;
			else
				point_sdf[tId] = 100000.0f;
			point_normal[tId] = Coord(0);
			return;
		}

		int id_0 = 0;
		VoxelOctreeNode<Coord> node_0;
		octree_node.getNode(pi, pj, pk, node_0, id_0);
		Coord pos_0 = node_0.position();

		Real coef = Real(pow(Real(2), int((node_0.level()))));
		Real maxdx_ = dx_ * coef;

		List<int>& node_list = octree_neighbors[id_0];
		int list_size = node_list.size();
		if (list_size == 0)
		{
			printf("~~~~~Error: The is no neighbor of that leaf node!~~~~~ %f %f %f %d %d %lld %d \n", origin_[0], origin_[1], origin_[2], id_0, node_0.value(), node_0.level(), node_0.midside());
		}

		Real dist = (pos_0 - pos).norm();
		Real weight;
		kernel3(weight, dist / maxdx_);
		//SO_InterpolationFunction(weight, pos, pos_0, maxdx_);
		Real weight_b = weight * octree_value[id_0];

		Vec4d b(1, pos_0[0], pos_0[1], pos_0[2]);
		b *= weight_b;

		Mat4d M(1.0f, pos_0[0], pos_0[1], pos_0[2],
			pos_0[0], pos_0[0] * pos_0[0], pos_0[0] * pos_0[1], pos_0[0] * pos_0[2],
			pos_0[1], pos_0[1] * pos_0[0], pos_0[1] * pos_0[1], pos_0[1] * pos_0[2],
			pos_0[2], pos_0[2] * pos_0[0], pos_0[2] * pos_0[1], pos_0[2] * pos_0[2]);
		M *= weight;

		for (int i = 0; i < list_size; i++)
		{
			Coord pos_i = octree_node[node_list[i]].position();

			dist = (pos_i - pos).norm();
			kernel3(weight, dist / maxdx_);
			//SO_InterpolationFunction(weight, pos, pos_i, maxdx_);
			weight_b = weight * octree_value[node_list[i]];

			Vec4d b_i(1, pos_i[0], pos_i[1], pos_i[2]);
			b += (b_i * weight_b);

			M(0, 0) += weight;				M(0, 1) += weight * pos_i[0];				M(0, 2) += weight * pos_i[1];				M(0, 3) += weight * pos_i[2];
			M(1, 0) += weight * pos_i[0];	M(1, 1) += weight * pos_i[0] * pos_i[0];	M(1, 2) += weight * pos_i[0] * pos_i[1];	M(1, 3) += weight * pos_i[0] * pos_i[2];
			M(2, 0) += weight * pos_i[1];	M(2, 1) += weight * pos_i[1] * pos_i[0];	M(2, 2) += weight * pos_i[1] * pos_i[1];	M(2, 3) += weight * pos_i[1] * pos_i[2];
			M(3, 0) += weight * pos_i[2];	M(3, 1) += weight * pos_i[2] * pos_i[0];	M(3, 2) += weight * pos_i[2] * pos_i[1];	M(3, 3) += weight * pos_i[2] * pos_i[2];
		}
		Mat4d M_inv = M.inverse();

		Vec4d x = M_inv * b;
		Vec4d p(1.0f, pos[0], pos[1], pos[2]);

		Real sdf_value = x.dot(p);
		if (abs(sdf_value) < 0.01*maxdx_) sdf_value = 0.0;

		Coord norm = Coord(x[1], x[2], x[3]);
		norm.normalize();
		if (inverted == true)
		{
			point_sdf[tId] = Real(-(sdf_value));
			point_normal[tId] = norm;
		}
		else
		{
			point_sdf[tId] = Real(sdf_value);
			point_normal[tId] = -norm;
		}
	}

	template<typename TDataType>
	void VoxelOctree<TDataType>::getSignDistanceMLS(
		DArray<Coord> point_pos,
		DArray<Real>& point_sdf,
		DArray<Coord>& point_normal,
		bool inverted)
	{
		point_sdf.resize(point_pos.size());
		point_normal.resize(point_pos.size());

		auto& neighbors = this->getNeighbors();

		int nx, ny, nz;
		this->getGrid(nx, ny, nz);

		cuExecute(point_pos.size(),
			SO_GetSignDistanceMLS,
			point_sdf,
			point_normal,
			point_pos,
			*this,
			neighbors,
			this->getSdfValues(),
			this->getOrigin(),
			this->getDx(),
			nx,
			ny,
			nz,
			inverted);
	}


	DEFINE_CLASS(VoxelOctree);
}