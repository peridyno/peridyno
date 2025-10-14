#include "FastSweepingMethodGPU.h"

#include "Algorithm/Reduction.h"

#include "Collision/Distance3D.h"

#include "LevelSetConstructionAndBooleanHelper.h"

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

	template<typename TDataType>
	void FastSweepingMethodGPU<TDataType>::compute()
	{
		this->makeLevelSet();
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

		Real dx = this->varSpacing()->getValue();
		uint padding = this->varPadding()->getValue();

		DArray3D<GridType> gridType;
		DArray3D<int> closestTriId;
		LevelSetConstructionAndBooleanHelper<TDataType>::initialFromTriangle(
			ts,
			dx,
			padding,
			sdf,
			origin,
			gridType,
			closestTriId);

		auto& phi = sdf.distances();
		ni = phi.nx();
		nj = phi.ny();
		nk = phi.nz();

		auto& vertices = ts->getPoints();
		auto& triangles = ts->triangleIndices();

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

		//counter3d.clear();
		//counter1d.clear();
		gridType.clear();
		closestTriId.clear();
		//neighbors.clear();
		//edgeNormal.clear();
		//vertexNormal.clear();
	}

	DEFINE_CLASS(FastSweepingMethodGPU);
}