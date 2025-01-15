#include "FastSweepingMethodGPU.h"

#include "Algorithm/Reduction.h"

#include "Collision/Distance3D.h"

namespace dyno
{
	IMPLEMENT_TCLASS(FastSweepingMethodGPU, TDataType)

	template<typename TDataType>
	FastSweepingMethodGPU<TDataType>::FastSweepingMethodGPU()
		: ComputeModule()
	{
	}

	template<typename TDataType>
	FastSweepingMethodGPU<TDataType>::~FastSweepingMethodGPU()
	{
	}

	enum GridType
	{
		Accepted = 0,
		Tentative = 1,
		Infinite = 2
	};

	template<typename TDataType>
	void FastSweepingMethodGPU<TDataType>::compute()
	{
		this->makeLevelSet();
	}

#define TRI_MIN(x, y, z) glm::min(x, glm::min(y, z))
#define TRI_MAX(x, y, z) glm::max(x, glm::max(y, z))

	template<typename Real>
	__global__ void FSM_AssignInifity(
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
	__global__ void FSM_FindNeighboringGrids(
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

	template<typename Real, typename Coord>
	__global__ void FSM_StoreTriangleIds(
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

	__global__ void FSM_Array3D_To_Array1d(
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

	template<typename Real, typename Coord, typename Tri2Edg>
	__global__ void FSM_InitializeSDFNearMesh(
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

			bool valid = calculateSignedDistance2TriangleSetFromNormal(p3d, gx, vertices, edges, indices, t2e, edgeN, vertexN, list);
			if (valid) {
				phi(i, j, k) = p3d.signed_distance;
				closestId(i, j, k) = p3d.id;
				gridType(i, j, k) = GridType::Accepted;
			}
		}
	}

	template<typename Real, typename Coord>
	GPU_FUNC void FSM_CheckNeighbour(
		DArray3D<Real>& phi,
		DArray3D<int>& closest_tri,
		const DArray<Coord>& vert,
		const DArray<TopologyModule::Triangle>& tri, 
		const Coord gx,
		int i0, int j0, int k0, 
		int i1, int j1, int k1)
	{
		if (closest_tri(i1, j1, k1) >= 0) {
			unsigned int p, q, r;
			auto trijk = tri[closest_tri(i1, j1, k1)];
			p = trijk[0];
			q = trijk[1];
			r = trijk[2];

			auto t = TTriangle3D<Real>(vert[p], vert[q], vert[r]);
			Real d = TPoint3D<Real>(gx).distance(t);

			Real phi0 = phi(i0, j0, k0);
			Real phi1 = phi(i1, j1, k1);

			if (glm::abs(d) < glm::abs(phi0)) {
				phi(i0, j0, k0) = phi1 > 0 ? glm::abs(d) : -glm::abs(d);
				closest_tri(i0, j0, k0) = closest_tri(i1, j1, k1);
			}
		}
	}

	template<typename Real, typename Coord>
	__global__	void FSM_SweepX(
		DArray3D<Real> phi,
		DArray3D<int> closestId,
		DArray3D<GridType> gridType, 
		DArray<Coord> vertices,
		DArray<TopologyModule::Triangle> indices,
		Coord origin,
		Real dx,
		int di)
	{
		int j = threadIdx.x + (blockIdx.x * blockDim.x);
		int k = threadIdx.y + (blockIdx.y * blockDim.y);

		uint ny = phi.ny();
		uint nz = phi.nz();

		if (j >= ny || k >= nz) return;

		int i0 = di > 0 ? 1 : phi.nx() - 2;
		int i1 = di > 0 ? phi.nx() : -1;

		for (int i = i0; i != i1; i += di) {
			if (gridType(i, j, k) != GridType::Accepted)
			{
				Coord gx(i * dx + origin[0], j * dx + origin[1], k * dx + origin[2]);
				FSM_CheckNeighbour(phi, closestId, vertices, indices, gx, i, j, k, i - di, j, k);
			}
		}
	}

	template<typename Real, typename Coord>
	__global__	void FSM_SweepY(
		DArray3D<Real> phi,
		DArray3D<int> closestId,
		DArray3D<GridType> gridType,
		DArray<Coord> vertices,
		DArray<TopologyModule::Triangle> indices,
		Coord origin,
		Real dx,
		int dj)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int k = threadIdx.y + (blockIdx.y * blockDim.y);

		uint nx = phi.nx();
		uint nz = phi.nz();

		if (i >= nx || k >= nz) return;

		int j0 = dj > 0 ? 1 : phi.ny() - 2;
		int j1 = dj > 0 ? phi.ny() : -1;

		for (int j = j0; j != j1; j += dj) {
			if (gridType(i, j, k) != GridType::Accepted)
			{
				Coord gx(i * dx + origin[0], j * dx + origin[1], k * dx + origin[2]);
				FSM_CheckNeighbour(phi, closestId, vertices, indices, gx, i, j, k, i, j - dj, k);
			}
		}
	}

	template<typename Real, typename Coord>
	__global__	void FSM_SweepZ(
		DArray3D<Real> phi,
		DArray3D<int> closestId,
		DArray3D<GridType> gridType,
		DArray<Coord> vertices,
		DArray<TopologyModule::Triangle> indices,
		Coord origin,
		Real dx,
		int dk)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);

		uint nx = phi.nx();
		uint ny = phi.ny();

		if (i >= nx || j >= ny) return;

		int k0 = dk > 0 ? 1 : phi.nz() - 2;
		int k1 = dk > 0 ? phi.nz() : -1;

		for (int k = k0; k != k1; k += dk) {
			if (gridType(i, j, k) != GridType::Accepted)
			{
				Coord gx(i * dx + origin[0], j * dx + origin[1], k * dx + origin[2]);
				FSM_CheckNeighbour(phi, closestId, vertices, indices, gx, i, j, k, i, j, k - dk);
			}
		}
	}

	template<typename TDataType>
	void FastSweepingMethodGPU<TDataType>::makeLevelSet()
	{
		if (this->outLevelSet()->isEmpty()) {
			this->outLevelSet()->allocate();
		}

		DistanceField3D<TDataType>& sdf = this->outLevelSet()->getDataPtr()->getSDF();

		auto ts = this->inTriangleSet()->constDataPtr();

		DArray<Coord> edgeNormal, vertexNormal;
		ts->updateEdgeNormal(edgeNormal);
		ts->updateAngleWeightedVertexNormal(vertexNormal);

		ts->updateTriangle2Edge();
		auto& tri2edg = ts->getTriangle2Edge();

		auto& vertices = ts->getPoints();
		auto& edges = ts->getEdges();
		auto& triangles = ts->getTriangles();

		Reduction<Coord> reduce;
		Coord min_box = reduce.minimum(vertices.begin(), vertices.size());
		Coord max_box = reduce.maximum(vertices.begin(), vertices.size());

		Real dx = this->varSpacing()->getValue();
		uint padding = this->varPadding()->getValue();

		Coord unit(1, 1, 1);
		min_box -= padding * dx * unit;
		max_box += padding * dx * unit;

		origin = min_box;
		maxPoint = max_box;

		sdf.setSpace(min_box, max_box, dx);
		auto& phi = sdf.distances();

		ni = phi.nx();
		nj = phi.ny();
		nk = phi.nz();

		//initialize distances near the mesh
		DArray3D<uint> counter3d(ni, nj, nk);
		DArray<uint> counter1d(ni * nj * nk);
		counter3d.reset();

		DArray3D<GridType> gridType(ni, nj, nk);
		DArray3D<int> closestTriId(ni, nj, nk);
		
		cuExecute3D(make_uint3(ni, nj, nk),
			FSM_AssignInifity,
			phi,
			closestTriId,
			gridType);

		cuExecute(triangles.size(),
			FSM_FindNeighboringGrids,
			counter3d,
			vertices,
			triangles,
			origin,
			dx);

		cuExecute3D(make_uint3(ni, nj, nk),
			FSM_Array3D_To_Array1d,
			counter1d,
			counter3d);

		DArrayList<uint> neighbors;
		neighbors.resize(counter1d);

		cuExecute(triangles.size(),
			FSM_StoreTriangleIds,
			neighbors,
			counter3d,
			vertices,
			triangles,
			origin,
			dx);

		cuExecute3D(make_uint3(ni, nj, nk),
			FSM_InitializeSDFNearMesh,
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

		// fill in the rest of the distances with fast sweeping
		uint passMax = this->varPassNumber()->getValue();
		for (unsigned int pass = 0; pass < passMax; ++pass) {

			// +x direction
			cuExecute2D(make_uint2(nj, nk),
				FSM_SweepX,
				phi,
				closestTriId,
				gridType,
				vertices,
				triangles,
				origin,
				dx,
				1);

			// -x direction
			cuExecute2D(make_uint2(nj, nk),
				FSM_SweepX,
				phi,
				closestTriId,
				gridType,
				vertices,
				triangles,
				origin,
				dx,
				-1);

			// +y direction
			cuExecute2D(make_uint2(ni, nk),
				FSM_SweepY,
				phi,
				closestTriId,
				gridType,
				vertices,
				triangles,
				origin,
				dx,
				1);

			// -y direction
			cuExecute2D(make_uint2(ni, nk),
				FSM_SweepY,
				phi,
				closestTriId,
				gridType,
				vertices,
				triangles,
				origin,
				dx,
				-1);

			// +z direction
			cuExecute2D(make_uint2(ni, nj),
				FSM_SweepZ,
				phi,
				closestTriId,
				gridType,
				vertices,
				triangles,
				origin,
				dx,
				1);

			// -z direction
			cuExecute2D(make_uint2(ni, nj),
				FSM_SweepZ,
				phi,
				closestTriId,
				gridType,
				vertices,
				triangles,
				origin,
				dx,
				-1);
		}

		counter3d.clear();
		counter1d.clear();
		gridType.clear();
		neighbors.clear();
		edgeNormal.clear();
		vertexNormal.clear();
	}

	DEFINE_CLASS(FastSweepingMethodGPU);
}