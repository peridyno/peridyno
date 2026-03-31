#include "AdaptiveVolumeFromTriangleSDF.h"
#include <thrust/sort.h>
#include <ctime>
#include "Timer.h"
#include "Collision/Distance3D.h"
#include "Array/ArrayMap.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(AdaptiveVolumeFromTriangleSDF, TDataType)

	__constant__ int voffset[6][2] = {
		1, 0,
		3, 0, 
		1, 3, 
		5, 0, 
		1, 5, 
		3, 5
	};

	template<typename TDataType>
	AdaptiveVolumeFromTriangleSDF<TDataType>::AdaptiveVolumeFromTriangleSDF()
		: AdaptiveVolumeFromTriangle<TDataType>()
	{
	}

	template<typename TDataType>
	AdaptiveVolumeFromTriangleSDF<TDataType>::~AdaptiveVolumeFromTriangleSDF()
	{
		m_GridType.clear();
		m_nodes.clear();
		m_vertex_neighbor.clear();
		m_node2ver.clear();
	}

	template <typename TDataType>
	GPU_FUNC bool AFTSDF_VertexAccess(
		int& index,
		int& i_index,
		int& j_index,
		int& k_index,
		Level& lmax,
		AdaptiveGridSet<TDataType>& gridSet,
		DArray<int>& node2vertex,
		DArray<AdaptiveGridNode>& nodes,
		DArray<int>& vertex_neighbor)
	{
		int nindex;
		OcKey mindex = CalculateMortonCode(i_index, j_index, k_index);
		gridSet.accessRandomLeafs(nindex, mindex, lmax);

		OcIndex gnx, gny, gnz;
		RecoverFromMortonCode(nodes[nindex].m_morton << (3 * (lmax - nodes[nindex].m_level)), gnx, gny, gnz);
		if (i_index == gnx && j_index == gny && k_index == gnz)
		{
			index = node2vertex[8 * nindex];
			return true;
		}
		else
		{
			int dx_num = 1 << (lmax - nodes[nindex].m_level - 1);
			int vi = ((i_index - gnx) == dx_num ? 1 : 0);
			int vj = ((j_index - gny) == dx_num ? 1 : 0);
			int vk = ((k_index - gnz) == dx_num ? 1 : 0);

			if (i_index != (gnx + vi * dx_num) || j_index != (gny + vj * dx_num) || k_index != (gnz + vk * dx_num)) return false;

			int vindex = vi + vj * 2 + vk * 4;
			index = vertex_neighbor[6 * node2vertex[8 * nindex] + voffset[vindex - 1][0]];
			if (voffset[vindex - 1][1] == 0)
				return true;
			else
			{
				index = vertex_neighbor[6 * index + voffset[vindex - 1][1]];
				return true;
			}	
		}
	}

	template <typename Real, typename Coord, typename Triangle, typename TDataType>
	__global__ void AFTSDF_TriangleInitial(
		DArray<uint> count,
		DArray<Triangle> surf_triangles,
		DArray<Coord> surf_points,
		Coord origin_,
		Real dx_,
		Level lmax,
		int resolution,
		AdaptiveGridSet<TDataType> gridSet,
		DArray<int> node2vertex,
		DArray<AdaptiveGridNode> nodes,
		DArray<int> vertex_neighbor)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= surf_triangles.size()) return;

		int p = surf_triangles[tId][0];
		int q = surf_triangles[tId][1];
		int r = surf_triangles[tId][2];

		Coord fp = (surf_points[p] - origin_) / dx_;
		Coord fq = (surf_points[q] - origin_) / dx_;
		Coord fr = (surf_points[r] - origin_) / dx_;

		int extend_band = 1;
		int nx_hi = clamp(int(maximum(fp[0], maximum(fq[0], fr[0]))) + extend_band + 1, 0, resolution - 1);
		int ny_hi = clamp(int(maximum(fp[1], maximum(fq[1], fr[1]))) + extend_band + 1, 0, resolution - 1);
		int nz_hi = clamp(int(maximum(fp[2], maximum(fq[2], fr[2]))) + extend_band + 1, 0, resolution - 1);

		int nx_lo = clamp(int(minimum(fp[0], minimum(fq[0], fr[0]))) - extend_band, 0, resolution - 1);
		int ny_lo = clamp(int(minimum(fp[1], minimum(fq[1], fr[1]))) - extend_band, 0, resolution - 1);
		int nz_lo = clamp(int(minimum(fp[2], minimum(fq[2], fr[2]))) - extend_band, 0, resolution - 1);

		for (int k = nz_lo; k <= nz_hi; k++)for (int j = ny_lo; j <= ny_hi; j++)for (int i = nx_lo; i <= nx_hi; i++)
		{
			int nindex;
			OcKey mindex = CalculateMortonCode(i, j, k);
			gridSet.accessRandomLeafs(nindex, mindex, lmax);
			atomicAdd(&count[node2vertex[8 * nindex]], 1);
			atomicAdd(&count[node2vertex[8 * nindex + 1]], 1);
			atomicAdd(&count[node2vertex[8 * nindex + 2]], 1);
			atomicAdd(&count[node2vertex[8 * nindex + 3]], 1);
			atomicAdd(&count[node2vertex[8 * nindex + 4]], 1);
			atomicAdd(&count[node2vertex[8 * nindex + 5]], 1);
			atomicAdd(&count[node2vertex[8 * nindex + 6]], 1);
			atomicAdd(&count[node2vertex[8 * nindex + 7]], 1);
		}
	}

	template <typename Real, typename Coord, typename Triangle, typename TDataType>
	__global__ void AFTSDF_TriangleCompute(
		DArrayList<int> tri_index,
		DArray<Triangle> surf_triangles,
		DArray<Coord> surf_points,
		Coord origin_,
		Real dx_,
		Level lmax,
		int resolution,
		AdaptiveGridSet<TDataType> gridSet,
		DArray<int> node2vertex,
		DArray<AdaptiveGridNode> nodes,
		DArray<int> vertex_neighbor)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= surf_triangles.size()) return;

		int p = surf_triangles[tId][0];
		int q = surf_triangles[tId][1];
		int r = surf_triangles[tId][2];

		Coord fp = (surf_points[p] - origin_) / dx_;
		Coord fq = (surf_points[q] - origin_) / dx_;
		Coord fr = (surf_points[r] - origin_) / dx_;

		int extend_band = 1;
		int nx_hi = clamp(int(maximum(fp[0], maximum(fq[0], fr[0]))) + extend_band + 1, 0, resolution - 1);
		int ny_hi = clamp(int(maximum(fp[1], maximum(fq[1], fr[1]))) + extend_band + 1, 0, resolution - 1);
		int nz_hi = clamp(int(maximum(fp[2], maximum(fq[2], fr[2]))) + extend_band + 1, 0, resolution - 1);

		int nx_lo = clamp(int(minimum(fp[0], minimum(fq[0], fr[0]))) - extend_band, 0, resolution - 1);
		int ny_lo = clamp(int(minimum(fp[1], minimum(fq[1], fr[1]))) - extend_band, 0, resolution - 1);
		int nz_lo = clamp(int(minimum(fp[2], minimum(fq[2], fr[2]))) - extend_band, 0, resolution - 1);

		for (int k = nz_lo; k <= nz_hi; k++) for (int j = ny_lo; j <= ny_hi; j++) for (int i = nx_lo; i <= nx_hi; i++)
		{
			int nindex;
			OcKey mindex = CalculateMortonCode(i, j, k);
			gridSet.accessRandomLeafs(nindex, mindex, lmax);
			tri_index[node2vertex[8 * nindex]].atomicInsert(tId);
			tri_index[node2vertex[8 * nindex + 1]].atomicInsert(tId);
			tri_index[node2vertex[8 * nindex + 2]].atomicInsert(tId);
			tri_index[node2vertex[8 * nindex + 3]].atomicInsert(tId);
			tri_index[node2vertex[8 * nindex + 4]].atomicInsert(tId);
			tri_index[node2vertex[8 * nindex + 5]].atomicInsert(tId);
			tri_index[node2vertex[8 * nindex + 6]].atomicInsert(tId);
			tri_index[node2vertex[8 * nindex + 7]].atomicInsert(tId);
		}
	}


	__global__ void AFTSDF_TringleListToMap(
		DArrayMap<bool> nmap,
		DArrayList<int> nlist,
		int vertex_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertex_num) return;

		if (nlist[tId].size() > 0)
		{
			for (int i = 0; i < nlist[tId].size(); i++)
				nmap[tId].insert(Pair<int, bool>(nlist[tId][i], true));
		}
	}

	template<typename Real, typename Coord, typename Tri2Edg>
	__global__ void AFTSDF_InitializeSDF(
		DArray<Real> phi,
		DArray<GridType> gridType,
		DArrayMap<bool> triIds,
		DArrayList<int> triIds_list,
		DArray<Coord> points,
		DArray<Topology::Triangle> indices,
		DArray<Topology::Edge> edges,
		DArray<Tri2Edg> t2e,
		DArray<Coord> edgeN,
		DArray<Coord> pointN,
		Coord origin,
		Real dx,
		DArray<Coord> vertexs)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= phi.size()) return;

		auto& map = triIds[tId];
		if (map.size() == 0)
		{
			phi[tId] = MAX_DISTANCE;
			gridType[tId] = GridType::Infinite;
			return;
		}
		auto& list = triIds_list[tId];
		list.clear();
		for (int i = 0; i < map.size(); i++) list.insert(map[i].first);

		ProjectedPoint3D<Real> p3d;
		bool valid = calculateSignedDistance2TriangleSetFromNormal(p3d, vertexs[tId], points, edges, indices, t2e, edgeN, pointN, list);
		if (valid)
		{
			phi[tId] = p3d.signed_distance;
			gridType[tId] = GridType::Accepted;
		}
	}

	template<typename TDataType>
	void AdaptiveVolumeFromTriangleSDF<TDataType>::initialSDF()
	{
		auto triSet = this->inTriangleSet()->getDataPtr();
		auto& triangles = triSet->triangleIndices();
		auto& points = triSet->getPoints();
		auto& edges = triSet->edgeIndices();
		DArray<Coord> edgeNormal, pointsNormal;
		triSet->requestEdgeNormals(edgeNormal);
		triSet->requestVertexNormals(pointsNormal);
		auto& tri2edg = triSet->triangle2Edge();

		auto m_AGrid = this->stateAGridSet()->getDataPtr();
		Coord m_origin = m_AGrid->adaptiveGridOrigin();
		Real m_dx = m_AGrid->adaptiveGridDx();
		Level m_levelmax = m_AGrid->adaptiveGridLevelMax();
		int resolution = (1 << m_levelmax);

		m_AGrid->extractLeafs(m_nodes);
		DArray<Coord> vertex;
		m_AGrid->extractVertex(vertex, m_vertex_neighbor, m_node2ver);

		auto& m_sdf = this->stateAGridSDF()->getData();
		m_sdf.resize(vertex.size());
		DArray<uint> count(m_sdf.size());
		count.reset();
		cuExecute(triangles.size(),
			AFTSDF_TriangleInitial,
			count,
			triangles,
			points,
			m_origin,
			m_dx,
			m_levelmax,
			resolution,
			*m_AGrid,
			m_node2ver,
			m_nodes,
			m_vertex_neighbor);

		DArrayList<int> tri_index;
		tri_index.resize(count);
		cuExecute(triangles.size(),
			AFTSDF_TriangleCompute,
			tri_index,
			triangles,
			points,
			m_origin,
			m_dx,
			m_levelmax,
			resolution,
			*m_AGrid,
			m_node2ver,
			m_nodes,
			m_vertex_neighbor);

		DArrayMap<bool> tri_map;
		tri_map.resize(count);
		tri_map.reset();
		cuExecute(m_sdf.size(),
			AFTSDF_TringleListToMap,
			tri_map,
			tri_index,
			m_sdf.size());

		m_GridType.resize(m_sdf.size());
		cuExecute(m_sdf.size(),
			AFTSDF_InitializeSDF,
			m_sdf,
			m_GridType,
			tri_map,
			tri_index,
			points,
			triangles,
			edges,
			tri2edg,
			edgeNormal,
			pointsNormal,
			m_origin,
			m_dx,
			vertex);

		vertex.clear();
		edgeNormal.clear();
		pointsNormal.clear();
		count.clear();
		tri_index.clear();
		tri_map.clear();
	}

	template<typename Real>
	__global__ void AFTSDF_ComputeSpace(
		DArray<Real> space,
		DArray<AdaptiveGridNode> nodes,
		DArray<int> node2ver,
		DArray<int> vneighbor,
		Real dx,
		Level lmax)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= (nodes.size())) return;

		//neighbor order is: -x,+x,-y,+y,-z,+z
		//vertex order is:(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1);(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)
		//if (nodes[tId].isLeaf())
		//{
		Real up_dx = dx * (1 << (lmax - (nodes[tId].m_level)));

		int v[8];
		v[0] = node2ver[8 * tId];
		v[1] = node2ver[8 * tId + 1];
		v[2] = node2ver[8 * tId + 2];
		v[3] = node2ver[8 * tId + 3];
		v[4] = node2ver[8 * tId + 4];
		v[5] = node2ver[8 * tId + 5];
		v[6] = node2ver[8 * tId + 6];
		v[7] = node2ver[8 * tId + 7];

		if (vneighbor[6 * v[0] + 1] == v[1])
		{
			space[6 * v[0] + 1] = up_dx;
			space[6 * v[1] + 0] = up_dx;
		}
		if (vneighbor[6 * v[0] + 3] == v[3])
		{
			space[6 * v[0] + 3] = up_dx;
			space[6 * v[3] + 2] = up_dx;
		}
		if (vneighbor[6 * v[0] + 5] == v[4])
		{
			space[6 * v[0] + 5] = up_dx;
			space[6 * v[4] + 4] = up_dx;
		}
		if (vneighbor[6 * v[1] + 3] == v[2])
		{
			space[6 * v[1] + 3] = up_dx;
			space[6 * v[2] + 2] = up_dx;
		}
		if (vneighbor[6 * v[1] + 5] == v[5])
		{
			space[6 * v[1] + 5] = up_dx;
			space[6 * v[5] + 4] = up_dx;
		}
		if (vneighbor[6 * v[2] + 0] == v[3])
		{
			space[6 * v[2] + 0] = up_dx;
			space[6 * v[3] + 1] = up_dx;
		}
		if (vneighbor[6 * v[2] + 5] == v[6])
		{
			space[6 * v[2] + 5] = up_dx;
			space[6 * v[6] + 4] = up_dx;
		}
		if (vneighbor[6 * v[3] + 5] == v[7])
		{
			space[6 * v[3] + 5] = up_dx;
			space[6 * v[7] + 4] = up_dx;
		}
		if (vneighbor[6 * v[4] + 1] == v[5])
		{
			space[6 * v[4] + 1] = up_dx;
			space[6 * v[5] + 0] = up_dx;
		}
		if (vneighbor[6 * v[4] + 3] == v[7])
		{
			space[6 * v[4] + 3] = up_dx;
			space[6 * v[7] + 2] = up_dx;
		}
		if (vneighbor[6 * v[5] + 3] == v[6])
		{
			space[6 * v[5] + 3] = up_dx;
			space[6 * v[6] + 2] = up_dx;
		}
		if (vneighbor[6 * v[6] + 0] == v[7])
		{
			space[6 * v[6] + 0] = up_dx;
			space[6 * v[7] + 1] = up_dx;
		}
		//}
	}

	GPU_FUNC void AFTSDF_Swap(
		Real& a,
		Real& b)
	{
		Real tmp = b;
		b = a;
		a = tmp;
	}

	template<typename Real>
	GPU_FUNC void AFTSDF_UpdatePhi(
		DArray<Real>& sdf,
		DArray<GridType>& type,
		DArray<int>& vneighbor,
		DArray<Real>& space,
		//DArray<Coord>& vertex,
		int index)
	{
		auto alpha = [&](int _ni, int _nj, Real& _sdf, Real& _space) -> void {
			int side1 = vneighbor[6 * index + _ni];
			int side2 = vneighbor[6 * index + _nj];
			if (side1 == EMPTY)
			{
				_sdf = sdf[side2];
				_space = space[6 * index + _nj];
				return;
			}
			if (side2 == EMPTY)
			{
				_sdf = sdf[side1];
				_space = space[6 * index + _ni];
				return;
			}
			if (abs(sdf[side1]) == min(abs(sdf[side1]), abs(sdf[side2])))
			{
				_sdf = sdf[side1];
				_space = space[6 * index + _ni];
				return;
			}
			else
			{
				_sdf = sdf[side2];
				_space = space[6 * index + _nj];
				return;
			}
			};

		Real phi_minx, phi_miny, phi_minz;
		Real space_x, space_y, space_z;
		alpha(0, 1, phi_minx, space_x);
		alpha(2, 3, phi_miny, space_y);
		alpha(4, 5, phi_minz, space_z);

		bool outside = true;
		if (phi_minx < -0.001 * space_x || phi_miny < -0.001 * space_y || phi_minz < -0.001 * space_z) outside = false;
		Real sign = outside ? Real(1) : Real(-1);

		phi_minx = (sign * phi_minx) > 0 ? phi_minx : (-phi_minx);
		phi_miny = (sign * phi_miny) > 0 ? phi_miny : (-phi_miny);
		phi_minz = (sign * phi_minz) > 0 ? phi_minz : (-phi_minz);

		Real a[3], b[3];
		a[0] = phi_minx;
		a[1] = phi_miny;
		a[2] = phi_minz;

		// Sort
		if (outside)
		{//a[0]<a[1]<a[2]
			if (a[0] > a[1]) AFTSDF_Swap(a[0], a[1]);
			if (a[1] > a[2]) AFTSDF_Swap(a[1], a[2]);
			if (a[0] > a[1]) AFTSDF_Swap(a[0], a[1]);
		}
		else
		{//a[2]<a[1]<a[0]
			if (a[0] < a[1]) AFTSDF_Swap(a[0], a[1]);
			if (a[1] < a[2]) AFTSDF_Swap(a[1], a[2]);
			if (a[0] < a[1]) AFTSDF_Swap(a[0], a[1]);
		}

		if (abs(a[0]) > 0.8 * MAX_DISTANCE) return;

		if (phi_minx == a[0])
		{
			b[0] = space_x;
			if (phi_miny == a[1])
			{
				b[1] = space_y;
				b[2] = space_z;
			}
			else if (phi_minz == a[1])
			{
				b[1] = space_z;
				b[2] = space_y;
			}
		}
		else if (phi_miny == a[0])
		{
			b[0] = space_y;
			if (phi_minx == a[1])
			{
				b[1] = space_x;
				b[2] = space_z;
			}
			else if (phi_minz == a[1])
			{
				b[1] = space_z;
				b[2] = space_x;
			}
		}
		else if (phi_minz == a[0])
		{
			b[0] = space_z;
			if (phi_minx == a[1])
			{
				b[1] = space_x;
				b[2] = space_y;
			}
			else if (phi_miny == a[1])
			{
				b[1] = space_y;
				b[2] = space_x;
			}
		}

		Real phi_ijk, dx = max(b[0], max(b[1], b[2]));

		if (glm::abs(a[0] - a[2]) < dx)
		{
			Real part_a = b[0] * b[0] * b[1] * b[1] + b[0] * b[0] * b[2] * b[2] + b[1] * b[1] * b[2] * b[2];
			Real part_b = a[2] * b[0] * b[0] * b[1] * b[1] + a[1] * b[0] * b[0] * b[2] * b[2] + a[0] * b[1] * b[1] * b[2] * b[2];
			Real part_c = a[2] * a[2] * b[0] * b[0] * b[1] * b[1] + a[1] * a[1] * b[0] * b[0] * b[2] * b[2] + a[0] * a[0] * b[1] * b[1] * b[2] * b[2] - b[0] * b[0] * b[1] * b[1] * b[2] * b[2];
			phi_ijk = (part_b + sign * glm::sqrt(abs(part_b * part_b - part_a * part_c))) / (part_a);
		}
		else if (glm::abs(a[0] - a[1]) < dx)
		{
			Real part_a = b[0] * b[0] + b[1] * b[1];
			Real part_b = a[1] * b[0] * b[0] + a[0] * b[1] * b[1];
			Real part_c = a[1] * a[1] * b[0] * b[0] + a[0] * a[0] * b[1] * b[1] - b[0] * b[0] * b[1] * b[1];
			phi_ijk = (part_b + sign * glm::sqrt(abs(part_b * part_b - part_a * part_c))) / (part_a);
		}
		else
		{
			phi_ijk = a[0] + sign * b[0];
		}

		//Real phi_ijk_old = sdf[index];
		//if (abs(phi_ijk) < abs(sdf[index]))
		sdf[index] = phi_ijk;
	}

	template<typename Real>
	__global__ void AFTSDF_FastIterative(
		DArray<Real> sdf,
		DArray<GridType> type,
		DArray<int> vneighbor,
		DArray<Real> space)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= (sdf.size())) return;

		if (type[tId] == GridType::Accepted) return;

		AFTSDF_UpdatePhi(sdf, type, vneighbor, space, tId);
	}

	template<typename TDataType>
	void AdaptiveVolumeFromTriangleSDF<TDataType>::computeSDF()
	{
		auto m_AGrid = this->stateAGridSet()->getDataPtr();
		Real m_dx = m_AGrid->adaptiveGridDx();
		Level m_levelmax = m_AGrid->adaptiveGridLevelMax();

		DArray<Real> space(m_vertex_neighbor.size());
		cuExecute(m_nodes.size(),
			AFTSDF_ComputeSpace,
			space,
			m_nodes,
			m_node2ver,
			m_vertex_neighbor,
			m_dx,
			m_levelmax);

		auto& m_sdf = this->stateAGridSDF()->getData();
		for (uint t = 0; t < 2 * m_levelmax; t++)
		{
			printf("~~~~~~~~~~~~~~~~Node iterative %d~~~~~~~~~~~ \n", t);

			cuExecute(m_sdf.size(),
				AFTSDF_FastIterative,
				m_sdf,
				m_GridType,
				m_vertex_neighbor,
				space);
		}

		space.clear();
	}

	template<typename TDataType>
	void AdaptiveVolumeFromTriangleSDF<TDataType>::resetStates()
	{
		AdaptiveVolumeFromTriangle<TDataType>::resetStates();

		initialSDF();
		computeSDF();
	}

	template<typename TDataType>
	void AdaptiveVolumeFromTriangleSDF<TDataType>::updateStates()
	{
		initialSDF();
		computeSDF();
	}

	DEFINE_CLASS(AdaptiveVolumeFromTriangleSDF);
}