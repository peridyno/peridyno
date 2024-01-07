#include "Wake.h"

#include "Math/Lerp.h"

#include "Algorithm/CudaRand.h"

namespace dyno
{
	IMPLEMENT_TCLASS(Wake, TDataType)

	template<typename TDataType>
	Wake<TDataType>::Wake()
		: CapillaryWave<TDataType>()
	{
	}

	template<typename TDataType>
	Wake<TDataType>::~Wake()
	{
	}

	template <typename Real, typename Coord2D, typename Coord3D, typename Coord4D, typename TriangleIndex>
	__global__ void W_AccumlateTrails(
		DArray2D<Coord2D> sources,
		DArray2D<Real> weights,
		DArray2D<Coord4D> grid,
		DArray<Coord3D> vertices,
		DArray<TriangleIndex> indices,
		Coord3D waveOrigin,
		Coord3D vesselCenter,
		Coord3D vesselVelocity,
		Coord3D vesselAngularVelocity,
		Real spacing)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= indices.size()) return;

		TriangleIndex index_i = indices[tId];

		Coord3D v0 = vertices[index_i[0]];
		Coord3D v1 = vertices[index_i[1]];
		Coord3D v2 = vertices[index_i[2]];

		Coord3D tc = (v0 + v1 + v2) / Real(3);

		uint nx = grid.nx();
		uint ny = grid.ny();

		Real x = (tc.x - waveOrigin.x) / spacing;
		Real y = (tc.z - waveOrigin.z) / spacing;

		Coord3D vel = vesselVelocity + (tc - vesselCenter).cross(vesselAngularVelocity);

		Coord4D g_i = bilinear(grid, x, y, LerpMode::CLAMP_TO_BORDER);

		//If the center is above the water surface, no impulse will be imposed
		Real d = (g_i.x - tc.y);
		d = d < 0 ? 0 : d;

		uint i0 = floor(x);
		uint j0 = floor(y);

		Real fx = x - i0;
		Real fy = y - j0;

		uint i1 = i0 + 1;
		uint j1 = j0 + 1;

		i0 = i0 < 1 ? 1 : (i0 >= nx - 2 ? nx - 2 : i0);
		j0 = j0 < 1 ? 1 : (j0 >= ny - 2 ? ny - 2 : j0);

		i1 = i1 < 1 ? 1 : (i1 >= nx - 2 ? nx - 2 : i1);
		j1 = j1 < 1 ? 1 : (j1 >= ny - 2 ? ny - 2 : j1);

		Real hu = d * vel.x;
		Real hv = d * vel.z;

		const float w00 = (1.0f - fx) * (1.0f - fy);
		const float w10 = fx * (1.0f - fy);
		const float w01 = (1.0f - fx) * fy;
		const float w11 = fx * fy;

		atomicAdd(&sources(i0, j0).x, w00 * hu);
		atomicAdd(&sources(i0, j0).y, w00 * hv);
		atomicAdd(&weights(i0, j0), w00);

		atomicAdd(&sources(i0, j1).x, w01 * hu);
		atomicAdd(&sources(i0, j1).y, w01 * hv);
		atomicAdd(&weights(i0, j1), w01);

		atomicAdd(&sources(i1, j0).x, w10 * hu);
		atomicAdd(&sources(i1, j0).y, w10 * hv);
		atomicAdd(&weights(i1, j0), w10);

		atomicAdd(&sources(i1, j1).x, w11 * hu);
		atomicAdd(&sources(i1, j1).y, w11 * hv);
		atomicAdd(&weights(i1, j1), w11);
	}

	template <typename Real, typename Coord2D, typename Coord4D>
	__global__ void W_AddTrails(
		DArray2D<Coord2D> sources,
		DArray2D<Real> weights,
		DArray2D<Coord4D> grid,
		Real mag)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		if (i < grid.nx() && j < grid.ny())
		{
			//if (weights(i, j) > 1)
			{
				RandNumber gen(i * j);

				auto rnd = gen.Generate();

				Coord4D gij = grid(i, j);
				Coord2D sij = weights(i, j) > 1 ? sources(i, j) / weights(i, j) : sources(i, j);

				gij.y += mag * rnd * sij.x;
				gij.z += mag * rnd * sij.y;

				grid(i, j) = gij;
			}
		}
	}

	template<typename TDataType>
	void Wake<TDataType>::addTrails()
	{
		uint res = this->varResolution()->getValue();

		Real dt = this->stateTimeStep()->getValue();

		auto heights = this->stateHeightField()->getDataPtr();
		auto& displacements = heights->getDisplacement();
		auto waveOrigin = heights->getOrigin();
		auto h = heights->getGridSpacing();

		{
			auto vessel = this->getVessel();
			auto& triangles = vessel->stateEnvelope()->getData();

			auto vesselCenter = vessel->stateCenter()->getData();
			auto vesselVelocity = vessel->stateVelocity()->getData();
			auto avesselAngularVelocity = vessel->stateAngularVelocity()->getData();

			auto& vertices = triangles.getPoints();
			auto& indices = triangles.getTriangles();

			uint num = indices.size();

			if (mDeviceGrid.nx() != mWeight.nx() || mDeviceGrid.ny() != mWeight.ny())
			{
				mWeight.resize(mDeviceGrid.nx(), mDeviceGrid.ny());
				mSource.resize(mDeviceGrid.nx(), mDeviceGrid.ny());
			}

			mWeight.reset();
			mSource.reset();

			cuExecute(num,
				W_AccumlateTrails,
				mSource,
				mWeight,
				mDeviceGrid,
				vertices,
				indices,
				waveOrigin,
				vesselCenter,
				vesselVelocity,
				avesselAngularVelocity,
				h);

			Real mag = this->varMagnitude()->getValue();

			cuExecute2D(make_uint2(mDeviceGrid.nx(), mDeviceGrid.ny()),
				W_AddTrails,
				mSource,
				mWeight,
				mDeviceGrid,
				mag);

			mDeviceGridNext.assign(mDeviceGrid);
		}
	}

	template<typename TDataType>
	void Wake<TDataType>::updateStates()
	{
		if (this->getVessel() != nullptr)
			addTrails();

		CapillaryWave<TDataType>::updateStates();
	}

	DEFINE_CLASS(Wake);
}