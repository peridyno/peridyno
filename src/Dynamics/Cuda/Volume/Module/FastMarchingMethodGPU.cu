#include "FastMarchingMethodGPU.h"

#include "Algorithm/Reduction.h"

#include "Collision/Distance3D.h"

#include "LevelSetConstructionAndBooleanHelper.h"

namespace dyno
{
	IMPLEMENT_TCLASS(FastMarchingMethodGPU, TDataType)

	template<typename TDataType>
	FastMarchingMethodGPU<TDataType>::FastMarchingMethodGPU()
		: ComputeModule()
	{
	}

	template<typename TDataType>
	FastMarchingMethodGPU<TDataType>::~FastMarchingMethodGPU()
	{
	}

	__global__	void FSMI_CheckTentativeX(
		DArray3D<GridType> pointType)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = pointType.nx();
		uint ny = pointType.ny();
		uint nz = pointType.nz();

		if (i >= nx - 1 || j >= ny || k >= nz) return;

		GridType type0 = pointType(i, j, k);
		GridType type1 = pointType(i + 1, j, k);

		if (type0 != GridType::Infinite && type1 == GridType::Infinite)
			pointType(i + 1, j, k) = GridType::Tentative;

		if (type0 == GridType::Infinite && type1 != GridType::Infinite)
			pointType(i, j, k) = GridType::Tentative;
	}

	__global__	void FSMI_CheckTentativeY(
		DArray3D<GridType> pointType)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = pointType.nx();
		uint ny = pointType.ny();
		uint nz = pointType.nz();

		if (i >= nx || j >= ny - 1 || k >= nz) return;

		GridType type0 = pointType(i, j, k);
		GridType type1 = pointType(i, j + 1, k);

		if (type0 != GridType::Infinite && type1 == GridType::Infinite)
			pointType(i, j + 1, k) = GridType::Tentative;

		if (type0 == GridType::Infinite && type1 != GridType::Infinite)
			pointType(i, j, k) = GridType::Tentative;
	}

	__global__	void FSMI_CheckTentativeZ(
		DArray3D<GridType> pointType)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = pointType.nx();
		uint ny = pointType.ny();
		uint nz = pointType.nz();

		if (i >= nx || j >= ny || k >= nz - 1) return;

		GridType type0 = pointType(i, j, k);
		GridType type1 = pointType(i, j, k + 1);

		if (type0 != GridType::Infinite && type1 == GridType::Infinite)
			pointType(i, j, k + 1) = GridType::Tentative;

		if (type0 == GridType::Infinite && type1 != GridType::Infinite)
			pointType(i, j, k) = GridType::Tentative;
	}

	template<typename TDataType>
	void FastMarchingMethodGPU<TDataType>::compute()
	{
		auto& inA = this->inLevelSetA()->getDataPtr()->getSDF();
		auto& inB = this->inLevelSetB()->getDataPtr()->getSDF();

		if (this->outLevelSet()->isEmpty()) {
			this->outLevelSet()->allocate();
		}

		DistanceField3D<TDataType>& out = this->outLevelSet()->getDataPtr()->getSDF();

		Real dx = this->varSpacing()->getValue();

		DArray3D<GridType> mGridType;
		DArray3D<bool> mOutside;
		LevelSetConstructionAndBooleanHelper<TDataType>::initialForBoolean(
			inA,
			inB,
			out,
			mGridType,
			mOutside,
			dx,
			1,
			this->varBoolType()->currentKey());

		auto& phi = out.distances();
		uint ni = phi.nx();
		uint nj = phi.ny();
		uint nk = phi.nz();

		DArray3D<uint> alpha(ni, nj, nk);
		for (uint t = 0; t < this->varMarchingNumber()->getValue(); t++)
		{
			// x direction
			cuExecute3D(make_uint3(ni - 1, nj, nk),
				FSMI_CheckTentativeX,
				mGridType);

			// y direction
			cuExecute3D(make_uint3(ni, nj - 1, nk),
				FSMI_CheckTentativeY,
				mGridType);

			// z direction
			cuExecute3D(make_uint3(ni, nj, nk - 1),
				FSMI_CheckTentativeZ,
				mGridType);

			LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
				phi,
				mGridType,
				alpha,
				mOutside,
				1,
				dx,
				false);
		}
		alpha.clear();
		mGridType.clear();
		mOutside.clear();
	}

	DEFINE_CLASS(FastMarchingMethodGPU);
}