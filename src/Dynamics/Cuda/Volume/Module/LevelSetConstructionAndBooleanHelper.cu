#include "LevelSetConstructionAndBooleanHelper.h"
#include "Algorithm/Reduction.h"
#include "MarchingCubesHelper.h"
//#include "Topology/EdgeSet.h"
//#include <thrust/sort.h>

namespace dyno
{
#define TRI_MIN(x, y, z) glm::min(x, glm::min(y, z))
#define TRI_MAX(x, y, z) glm::max(x, glm::max(y, z))

	template<typename Real>
	__global__ void LSCB_AssignInifity(
		DArray3D<Real> phi,
		DArray3D<int> closest_tri_id,
		DArray3D<GridType> types)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = phi.nx();
		uint ny = phi.ny();
		uint nz = phi.nz();

		if (i >= nx || j >= ny || k >= nz) return;

		phi(i, j, k) = REAL_MAX;
		closest_tri_id(i, j, k) = -1;
		types(i, j, k) = GridType::Infinite;
	}

	template<typename Real, typename Coord>
	__global__ void LSCB_FindNeighboringGrids(
		DArray3D<uint> counter,
		DArray<Coord> vertices,
		DArray<TopologyModule::Triangle> indices,
		Coord origin,
		Real dx)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= indices.size()) return;

		int ni = counter.nx();
		int nj = counter.ny();
		int nk = counter.nz();

		auto t = indices[tId];

		Coord vp = vertices[t[0]];
		Coord vq = vertices[t[1]];
		Coord vr = vertices[t[2]];

		// coordinates in grid to high precision
		Real fip = (vp[0] - origin[0]) / dx;
		Real fjp = (vp[1] - origin[1]) / dx;
		Real fkp = (vp[2] - origin[2]) / dx;
		Real fiq = (vq[0] - origin[0]) / dx;
		Real fjq = (vq[1] - origin[1]) / dx;
		Real fkq = (vq[2] - origin[2]) / dx;
		Real fir = (vr[0] - origin[0]) / dx;
		Real fjr = (vr[1] - origin[1]) / dx;
		Real fkr = (vr[2] - origin[2]) / dx;

		// do distances nearby
		const int exact_band = 1;

		int i0 = glm::clamp(int(TRI_MIN(fip, fiq, fir)) - exact_band, 0, ni - 1);
		int i1 = glm::clamp(int(TRI_MAX(fip, fiq, fir)) + exact_band + 1, 0, ni - 1);
		int j0 = glm::clamp(int(TRI_MIN(fjp, fjq, fjr)) - exact_band, 0, nj - 1);
		int j1 = glm::clamp(int(TRI_MAX(fjp, fjq, fjr)) + exact_band + 1, 0, nj - 1);
		int k0 = glm::clamp(int(TRI_MIN(fkp, fkq, fkr)) - exact_band, 0, nk - 1);
		int k1 = glm::clamp(int(TRI_MAX(fkp, fkq, fkr)) + exact_band + 1, 0, nk - 1);

		TTriangle3D<Real> tri(vp, vq, vr);
		for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) for (int i = i0; i <= i1; ++i) {
			TPoint3D<Real> point(origin + Coord(i * dx, j * dx, k * dx));
			if (glm::abs(point.distance(tri)) < dx * 1.74)	//1.74 is chosen to be slightly larger than sqrt(3)
			{
				atomicAdd(&counter(i, j, k), 1);
			}
		}
	}

	__global__ void LSCB_Array3D_To_Array1d(
		DArray<uint> arr1d,
		DArray3D<uint> arr3d)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = arr3d.nx();
		uint ny = arr3d.ny();
		uint nz = arr3d.nz();

		if (i >= nx || j >= ny || k >= nz) return;

		uint index1d = arr3d.index(i, j, k);

		arr1d[index1d] = arr3d(i, j, k);
	}

	template<typename Real, typename Coord>
	__global__ void LSCB_StoreTriangleIds(
		DArrayList<uint> triIds,
		DArray3D<uint> counter,
		DArray<Coord> vertices,
		DArray<TopologyModule::Triangle> indices,
		Coord origin,
		Real dx)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= indices.size()) return;

		int ni = counter.nx();
		int nj = counter.ny();
		int nk = counter.nz();

		auto t = indices[tId];

		Coord vp = vertices[t[0]];
		Coord vq = vertices[t[1]];
		Coord vr = vertices[t[2]];

		// coordinates in grid to high precision
		Real fip = (vp[0] - origin[0]) / dx;
		Real fjp = (vp[1] - origin[1]) / dx;
		Real fkp = (vp[2] - origin[2]) / dx;
		Real fiq = (vq[0] - origin[0]) / dx;
		Real fjq = (vq[1] - origin[1]) / dx;
		Real fkq = (vq[2] - origin[2]) / dx;
		Real fir = (vr[0] - origin[0]) / dx;
		Real fjr = (vr[1] - origin[1]) / dx;
		Real fkr = (vr[2] - origin[2]) / dx;

		// do distances nearby
		const int exact_band = 1;

		int i0 = glm::clamp(int(TRI_MIN(fip, fiq, fir)) - exact_band, 0, ni - 1);
		int i1 = glm::clamp(int(TRI_MAX(fip, fiq, fir)) + exact_band + 1, 0, ni - 1);
		int j0 = glm::clamp(int(TRI_MIN(fjp, fjq, fjr)) - exact_band, 0, nj - 1);
		int j1 = glm::clamp(int(TRI_MAX(fjp, fjq, fjr)) + exact_band + 1, 0, nj - 1);
		int k0 = glm::clamp(int(TRI_MIN(fkp, fkq, fkr)) - exact_band, 0, nk - 1);
		int k1 = glm::clamp(int(TRI_MAX(fkp, fkq, fkr)) + exact_band + 1, 0, nk - 1);

		TTriangle3D<Real> tri(vp, vq, vr);
		for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) for (int i = i0; i <= i1; ++i) {
			TPoint3D<Real> point(origin + Coord(i * dx, j * dx, k * dx));
			if (glm::abs(point.distance(tri)) < dx * 1.74)	//1.74 is chosen to be slightly larger than sqrt(3)
			{
				triIds[counter.index(i, j, k)].atomicInsert(tId);
			}
		}
	}

	template<typename Real, typename Coord, typename Tri2Edg>
	__global__ void LSCB_InitializeSDFNearMesh(
		DArray3D<Real> phi,
		DArray3D<int> closestId,
		DArray3D<GridType> gridType,
		DArrayList<uint> triIds,
		DArray<Coord> vertices,
		DArray<TopologyModule::Triangle> indices,
		DArray<TopologyModule::Edge> edges,
		DArray<Tri2Edg> t2e,
		DArray<Coord> edgeN,
		DArray<Coord> vertexN,
		Coord origin,
		Real dx)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = phi.nx();
		uint ny = phi.ny();
		uint nz = phi.nz();

		if (i >= nx || j >= ny || k >= nz) return;

		Coord gx(i * dx + origin[0], j * dx + origin[1], k * dx + origin[2]);

		auto& list = triIds[phi.index(i, j, k)];

		if (list.size() > 0)
		{
			ProjectedPoint3D<Real> p3d;

			TOrientedBox3D<Real> obb(Coord(0), Quat1f(), Coord(0.5));
			TPoint3D<Real> point(gx);
			Real exact_dist = point.distance(obb);

			bool valid = calculateSignedDistance2TriangleSetFromNormal(p3d, gx, vertices, edges, indices, t2e, edgeN, vertexN, list);
			//bool valid = calculateSignedDistance2TriangleSet(p3d, gx, vertices, indices, list);

			// 			if (abs(p3d.signed_distance - exact_dist) >= EPSILON)
			// 			{
			// 				printf("Error: %f %f %f; approximate: %f; exact: %f \n", gx.x, gx.y, gx.z, p3d.signed_distance, exact_dist);
			// 			}

			if (valid) {
				phi(i, j, k) = p3d.signed_distance;
				closestId(i, j, k) = p3d.id;
				gridType(i, j, k) = GridType::Accepted;
			}
		}
	}

	template<typename TDataType>
	void LevelSetConstructionAndBooleanHelper<TDataType>::initialFromTriangle(
		std::shared_ptr<TriangleSet<TDataType>> triSet,
		Real dx,
		uint padding,
		DistanceField3D<TDataType>& sdf,
		Coord& origin,
		DArray3D<GridType>& gridType,
		DArray3D<int>& closestTriId)
	{
		DArray<Coord> edgeNormal, vertexNormal;
		triSet->requestEdgeNormals(edgeNormal);
		triSet->requestVertexNormals(vertexNormal);

		auto& tri2edg = triSet->triangle2Edge();

		auto& vertices = triSet->getPoints();
		auto& edges = triSet->edgeIndices();
		auto& triangles = triSet->triangleIndices();

		Reduction<Coord> reduce;
		Coord min_box = reduce.minimum(vertices.begin(), vertices.size());
		Coord max_box = reduce.maximum(vertices.begin(), vertices.size());

		int min_i = std::floor(min_box[0] / dx) - padding;
		int min_j = std::floor(min_box[1] / dx) - padding;
		int min_k = std::floor(min_box[2] / dx) - padding;

		int max_i = std::ceil(max_box[0] / dx) + padding;
		int max_j = std::ceil(max_box[1] / dx) + padding;
		int max_k = std::ceil(max_box[2] / dx) + padding;

		// Use a background grid that is aligned with an interval of N times dx
		min_box = Coord(min_i * dx, min_j * dx, min_k * dx);
		max_box = Coord(max_i * dx, max_j * dx, max_k * dx);

		origin = min_box;

		sdf.setSpace(min_box, max_box, dx);
		auto& phi = sdf.distances();

		int ni = phi.nx();
		int nj = phi.ny();
		int nk = phi.nz();

		//initialize distances near the mesh
		DArray3D<uint> counter3d(ni, nj, nk);
		DArray<uint> counter1d(ni * nj * nk);
		counter3d.reset();

		gridType.resize(ni, nj, nk);
		closestTriId.resize(ni, nj, nk);

		cuExecute3D(make_uint3(ni, nj, nk),
			LSCB_AssignInifity,
			phi,
			closestTriId,
			gridType);

		cuExecute(triangles.size(),
			LSCB_FindNeighboringGrids,
			counter3d,
			vertices,
			triangles,
			origin,
			dx);

		cuExecute3D(make_uint3(ni, nj, nk),
			LSCB_Array3D_To_Array1d,
			counter1d,
			counter3d);

		DArrayList<uint> neighbors;
		neighbors.resize(counter1d);

		cuExecute(triangles.size(),
			LSCB_StoreTriangleIds,
			neighbors,
			counter3d,
			vertices,
			triangles,
			origin,
			dx);

		cuExecute3D(make_uint3(ni, nj, nk),
			LSCB_InitializeSDFNearMesh,
			phi,
			closestTriId,
			gridType,
			neighbors,
			vertices,
			triangles,
			edges,
			tri2edg,
			edgeNormal,
			vertexNormal,
			origin,
			dx);

		edgeNormal.clear();
		vertexNormal.clear();
		counter3d.clear();
		counter1d.clear();
		neighbors.clear();
	}


#define IndexInside( i, N ) \
    ((i) >= 0 && i < N)

	GPU_FUNC void MUTI_Swap(
		Real& a,
		Real& b)
	{
		Real tmp = b;
		b = a;
		a = tmp;
	}

	GPU_FUNC void LSCB_UpdatePhi(
		DArray3D<Real>& phi,
		DArray3D<GridType>& gridtype,
		DArray3D<bool>& outside,
		uint i, uint j, uint k,
		int interval,
		Real dx)
	{
		uint nx = phi.nx();
		uint ny = phi.ny();
		uint nz = phi.nz();

		bool outside_ijk = outside(i, j, k);

		Real phi_xmin, phi_ymin, phi_zmin;

		if (IndexInside(i - interval, nx) && IndexInside(i + interval, nx))
		{
			Real phi_minus = gridtype(i - interval, j, k) == GridType::Infinite ? (outside_ijk ? REAL_MAX : -REAL_MAX) : phi(i - interval, j, k);
			Real phi_plus = gridtype(i + interval, j, k) == GridType::Infinite ? (outside_ijk ? REAL_MAX : -REAL_MAX) : phi(i + interval, j, k);
			phi_xmin = outside_ijk ? minimum(phi_minus, phi_plus) : maximum(phi_minus, phi_plus);
		}
		else if (IndexInside(i - interval, nx) && !IndexInside(i + interval, nx))
		{
			Real phi_minus = gridtype(i - interval, j, k) == GridType::Infinite ? (outside_ijk ? REAL_MAX : -REAL_MAX) : phi(i - interval, j, k);
			phi_xmin = phi_minus;
		}
		else if (!IndexInside(i - interval, nx) && IndexInside(i + interval, nx))
		{
			Real phi_plus = gridtype(i + interval, j, k) == GridType::Infinite ? (outside_ijk ? REAL_MAX : -REAL_MAX) : phi(i + interval, j, k);
			phi_xmin = phi_plus;
		}
		else
			phi_xmin = outside_ijk ? REAL_MAX : -REAL_MAX;


		if (IndexInside(j - interval, ny) && IndexInside(j + interval, ny))
		{
			Real phi_minus = gridtype(i, j - interval, k) == GridType::Infinite ? (outside_ijk ? REAL_MAX : -REAL_MAX) : phi(i, j - interval, k);
			Real phi_plus = gridtype(i, j + interval, k) == GridType::Infinite ? (outside_ijk ? REAL_MAX : -REAL_MAX) : phi(i, j + interval, k);

			phi_ymin = outside_ijk ? minimum(phi_minus, phi_plus) : maximum(phi_minus, phi_plus);
		}
		else if (IndexInside(j - interval, ny) && !IndexInside(j + interval, ny))
		{
			Real phi_minus = gridtype(i, j - interval, k) == GridType::Infinite ? (outside_ijk ? REAL_MAX : -REAL_MAX) : phi(i, j - interval, k);
			phi_ymin = phi_minus;
		}
		else if (!IndexInside(j - interval, ny) && IndexInside(j + interval, ny))
		{
			Real phi_plus = gridtype(i, j + interval, k) == GridType::Infinite ? (outside_ijk ? REAL_MAX : -REAL_MAX) : phi(i, j + interval, k);
			phi_ymin = phi_plus;
		}
		else
			phi_ymin = outside_ijk ? REAL_MAX : -REAL_MAX;


		if (IndexInside(k - interval, nz) && IndexInside(k + interval, nz))
		{
			Real phi_minus = gridtype(i, j, k - interval) == GridType::Infinite ? (outside_ijk ? REAL_MAX : -REAL_MAX) : phi(i, j, k - interval);
			Real phi_plus = gridtype(i, j, k + interval) == GridType::Infinite ? (outside_ijk ? REAL_MAX : -REAL_MAX) : phi(i, j, k + interval);

			phi_zmin = outside_ijk ? minimum(phi_minus, phi_plus) : maximum(phi_minus, phi_plus);
		}
		else if (IndexInside(k - interval, nz) && !IndexInside(k + interval, nz))
		{
			Real phi_minus = gridtype(i, j, k - interval) == GridType::Infinite ? (outside_ijk ? REAL_MAX : -REAL_MAX) : phi(i, j, k - interval);
			phi_zmin = phi_minus;
		}
		else if (!IndexInside(k - interval, nz) && IndexInside(k + interval, nz))
		{
			Real phi_plus = gridtype(i, j, k + interval) == GridType::Infinite ? (outside_ijk ? REAL_MAX : -REAL_MAX) : phi(i, j, k + interval);
			phi_zmin = phi_plus;
		}
		else
			phi_zmin = outside_ijk ? REAL_MAX : -REAL_MAX;

		Real a[3];
		a[0] = phi_xmin;
		a[1] = phi_ymin;
		a[2] = phi_zmin;

		if (outside_ijk)
		{
			if (a[0] > a[1]) MUTI_Swap(a[0], a[1]);
			if (a[1] > a[2]) MUTI_Swap(a[1], a[2]);
			if (a[0] > a[1]) MUTI_Swap(a[0], a[1]);
		}
		else
		{
			if (a[0] < a[1]) MUTI_Swap(a[0], a[1]);
			if (a[1] < a[2]) MUTI_Swap(a[1], a[2]);
			if (a[0] < a[1]) MUTI_Swap(a[0], a[1]);
		}

		Real phi_ijk;
		Real h = interval * dx;
		Real sign = outside_ijk ? Real(1) : Real(-1);

		Real sum_a = a[0] + a[1] + a[2];
		Real sum_a2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];

		if (glm::abs(a[0] - a[2]) < h)
		{
			phi_ijk = (2 * sum_a + sign * glm::sqrt(4 * sum_a * sum_a - 12 * (sum_a2 - h * h))) / Real(6);
		}
		else if (glm::abs(a[0] - a[1]) < h)
		{
			phi_ijk = 0.5 * (a[0] + a[1] + sign * glm::sqrt(2 * h * h - (a[0] - a[1]) * (a[0] - a[1])));
		}
		else
		{
			phi_ijk = a[0] + sign * h;
		}

		phi(i, j, k) = phi_ijk;
	}

	template<typename Real>
	__global__ void LSCB_FastIterative(
		DArray3D<Real> phi,
		DArray3D<GridType> gridtype,
		DArray3D<uint> alpha,
		DArray3D<bool> outside,
		int interval,
		Real dx,
		bool controlInterval)
	{
		int i = (blockIdx.x * blockDim.x + threadIdx.x);
		int j = (blockIdx.y * blockDim.y + threadIdx.y);
		int k = (blockIdx.z * blockDim.z + threadIdx.z);

		uint nx = phi.nx();
		uint ny = phi.ny();
		uint nz = phi.nz();

		if (i >= nx || j >= ny || k >= nz) return;

		if (controlInterval == true)
		{
			if (gridtype(i, j, k) != GridType::Tentative || interval > alpha(i, j, k)) return;
		}
		else
		{
			if (gridtype(i, j, k) != GridType::Tentative) return;
		}
		
		LSCB_UpdatePhi(phi, gridtype, outside, i, j, k, interval, dx);
	}

	template<typename TDataType>
	void LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
		DArray3D<Real>& phi,
		DArray3D<GridType>& gridtype,
		DArray3D<uint>& alpha,
		DArray3D<bool>& outside,
		int interval,
		Real dx,
		bool controlInterval)
	{
		cuExecute3D(make_uint3(phi.nx(), phi.ny(), phi.nz()),
			LSCB_FastIterative,
			phi,
			gridtype,
			alpha,
			outside,
			interval,
			dx,
			controlInterval);
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void LSCB_BooleanOp(
		DArray3D<Real> distance,
		DArray3D<GridType> type,
		DArray3D<bool> outside,
		DistanceField3D<TDataType> fieldA,
		DistanceField3D<TDataType> fieldB,
		Coord origin,
		Real dx,
		int boolType)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = distance.nx();
		uint ny = distance.ny();
		uint nz = distance.nz();

		if (i >= nx || j >= ny || k >= nz) return;

		Coord point = origin + Coord(i * dx, j * dx, k * dx);

		Real a;
		Coord normal;
		fieldA.getDistance(point, a, normal);

		Real b;
		fieldB.getDistance(point, b, normal);

		Real iso = 0;

		Real op = FARWAY_DISTANCE;
		switch (boolType)
		{
		case 0://A intersect B
			op = a > b ? a : b;
			type(i, j, k) = (Inside(a, iso, dx) && Inside(b, iso, dx)) ? GridType::Accepted : GridType::Infinite;
			outside(i, j, k) = (Inside(a, iso, dx) && Inside(b, iso, dx)) ? false : true;
			break;
		case 1://A union B
			op = a > b ? b : a;
			type(i, j, k) = (Outside(a, iso, dx) && Outside(b, iso, dx) && op < 3.5 * dx) ? GridType::Accepted : GridType::Infinite;
			outside(i, j, k) = (Outside(a, iso, dx) && Outside(b, iso, dx)) ? true : false;
			break;
		case 2://A minus B
			op = a > -b ? a : -b;
			type(i, j, k) = (Inside(a, iso, dx) && Outside(b, iso, dx)) ? GridType::Accepted : GridType::Infinite;
			outside(i, j, k) = (Inside(a, iso, dx) && Outside(b, iso, dx)) ? false : true;
			break;
		default:
			break;
		}

		distance(i, j, k) = op;
	}

	template<typename TDataType>
	void LevelSetConstructionAndBooleanHelper<TDataType>::initialForBoolean(
		DistanceField3D<TDataType>& inA,
		DistanceField3D<TDataType>& inB,
		DistanceField3D<TDataType>& out,
		DArray3D<GridType>& gridType,
		DArray3D<bool>& outside,
		Real dx,
		uint padding,
		int boolType)
	{
		//Calculate the bounding box
		Coord min_box(std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max());
		Coord max_box(-std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max());

		min_box = min_box.minimum(inA.lowerBound());
		min_box = min_box.minimum(inB.lowerBound());

		max_box = max_box.maximum(inA.upperBound());
		max_box = max_box.maximum(inB.upperBound());

		// Use a background grid that is aligned with an interval of N times dx
		int min_i = std::floor(min_box[0] / dx) - padding;
		int min_j = std::floor(min_box[1] / dx) - padding;
		int min_k = std::floor(min_box[2] / dx) - padding;

		int max_i = std::ceil(max_box[0] / dx) + padding;
		int max_j = std::ceil(max_box[1] / dx) + padding;
		int max_k = std::ceil(max_box[2] / dx) + padding;

		min_box = Coord(min_i * dx, min_j * dx, min_k * dx);
		max_box = Coord(max_i * dx, max_j * dx, max_k * dx);

		out.setSpace(min_box, max_box, dx);
		auto& phi = out.distances();

		uint ni = phi.nx();
		uint nj = phi.ny();
		uint nk = phi.nz();

		gridType.resize(ni, nj, nk);
		outside.resize(ni, nj, nk);

		//Calculate the boolean of two distance fields
		cuExecute3D(make_uint3(ni, nj, nk),
			LSCB_BooleanOp,
			phi,
			gridType,
			outside,
			inA,
			inB,
			min_box,
			dx,
			boolType);
	}

	DEFINE_CLASS(LevelSetConstructionAndBooleanHelper);
}