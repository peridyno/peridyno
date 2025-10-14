//get run time
#include "MultiscaleFastIterativeMethodForBoolean.h"

#include "Algorithm/Reduction.h"

#include "Collision/Distance3D.h"

#include "Volume/BasicShapeToVolume.h"

#include "LevelSetConstructionAndBooleanHelper.h"

namespace dyno
{
	IMPLEMENT_TCLASS(MultiscaleFastIterativeMethodForBoolean, TDataType)

		template<typename TDataType>
	MultiscaleFastIterativeMethodForBoolean<TDataType>::MultiscaleFastIterativeMethodForBoolean()
		: ComputeModule()
	{
	}

	template<typename TDataType>
	MultiscaleFastIterativeMethodForBoolean<TDataType>::~MultiscaleFastIterativeMethodForBoolean()
	{
	}

	template<typename TDataType>
	void MultiscaleFastIterativeMethodForBoolean<TDataType>::compute()
	{
		this->makeLevelSet();
	}

	template<typename Real>
	__global__	void CheckTentative_Boolean_X(
		DArray3D<Real> phi,
		DArray3D<Real> phiBuffer,
		DArray3D<GridType> pointType,
		DArray3D<GridType> pointTypeBuffer,
		DArray3D<uint> alpha,
		DArray3D<bool> outside,
		int interval)
	{
		int i = (blockIdx.x * blockDim.x + threadIdx.x);
		int j = (blockIdx.y * blockDim.y + threadIdx.y);
		int k = (blockIdx.z * blockDim.z + threadIdx.z);

		uint nx = pointType.nx();
		uint ny = pointType.ny();
		uint nz = pointType.nz();

		if (i >= nx - interval || j >= ny || k >= nz) return;

		GridType type0 = pointTypeBuffer(i, j, k);
		GridType type1 = pointTypeBuffer(i + interval, j, k);

		if (type0 == GridType::Infinite && type1 != GridType::Infinite) {
			pointType(i, j, k) = GridType::Tentative;
			alpha(i, j, k) = interval;
			phi(i, j, k) = outside(i, j, k) ? REAL_MAX : -REAL_MAX;
		}

		if (type0 != GridType::Infinite && type1 == GridType::Infinite) {
			pointType(i + interval, j, k) = GridType::Tentative;
			alpha(i + interval, j, k) = interval;
			phi(i + interval, j, k) = outside(i + interval, j, k) ? REAL_MAX : -REAL_MAX;
		}
	}

	template<typename Real>
	__global__	void CheckTentative_Boolean_Y(
		DArray3D<Real> phi,
		DArray3D<GridType> pointType,
		DArray3D<GridType> pointTypeBuffer,
		DArray3D<uint> alpha,
		DArray3D<bool> outside,
		int interval)
	{
		int i = (blockIdx.x * blockDim.x + threadIdx.x);
		int j = (blockIdx.y * blockDim.y + threadIdx.y);
		int k = (blockIdx.z * blockDim.z + threadIdx.z);

		uint nx = pointType.nx();
		uint ny = pointType.ny();
		uint nz = pointType.nz();

		if (i >= nx || j >= ny - interval || k >= nz) return;

		GridType type0 = pointTypeBuffer(i, j, k);
		GridType type1 = pointTypeBuffer(i, j + interval, k);

		if (type0 == GridType::Infinite && type1 != GridType::Infinite) {
			pointType(i, j, k) = GridType::Tentative;
			alpha(i, j, k) = interval;
			phi(i, j, k) = outside(i, j, k) ? REAL_MAX : -REAL_MAX;
		}

		if (type0 != GridType::Infinite && type1 == GridType::Infinite) {
			pointType(i, j + interval, k) = GridType::Tentative;
			alpha(i, j + interval, k) = interval;
			phi(i, j + interval, k) = outside(i, j + interval, k) ? REAL_MAX : -REAL_MAX;
		}
	}

	template<typename Real>
	__global__	void CheckTentative_Boolean_Z(
		DArray3D<Real> phi,
		DArray3D<GridType> pointType,
		DArray3D<GridType> pointTypeBuffer,
		DArray3D<uint> alpha,
		DArray3D<bool> outside,
		int interval)
	{
		int i = (blockIdx.x * blockDim.x + threadIdx.x);
		int j = (blockIdx.y * blockDim.y + threadIdx.y);
		int k = (blockIdx.z * blockDim.z + threadIdx.z);

		uint nx = pointType.nx();
		uint ny = pointType.ny();
		uint nz = pointType.nz();

		if (i >= nx || j >= ny || k >= nz - interval) return;

		GridType type0 = pointTypeBuffer(i, j, k);
		GridType type1 = pointTypeBuffer(i, j, k + interval);

		if (type0 == GridType::Infinite && type1 != GridType::Infinite) {
			pointType(i, j, k) = GridType::Tentative;
			alpha(i, j, k) = interval;
			phi(i, j, k) = outside(i, j, k) ? REAL_MAX : -REAL_MAX;
		}

		if (type0 != GridType::Infinite && type1 == GridType::Infinite) {
			pointType(i, j, k + interval) = GridType::Tentative;
			alpha(i, j, k + interval) = interval;
			phi(i, j, k + interval) = outside(i, j, k + interval) ? REAL_MAX : -REAL_MAX;
		}
	}

	template<typename TDataType>
	void MultiscaleFastIterativeMethodForBoolean<TDataType>::makeLevelSet()
	{
		if (this->outLevelSet()->isEmpty()) {
			this->outLevelSet()->allocate();
		}

		auto& inA = this->inLevelSetA()->getDataPtr()->getSDF();
		auto& inB = this->inLevelSetB()->getDataPtr()->getSDF();

		DistanceField3D<TDataType>& out = this->outLevelSet()->getDataPtr()->getSDF();

		Real h = this->varSpacing()->getValue();
		uint padding = this->varPadding()->getValue();
		uint V_max = this->varVCircle()->getValue();


		DArray3D<GridType> mGridType;
		DArray3D<bool> mOutside;
		LevelSetConstructionAndBooleanHelper<TDataType>::initialForBoolean(
			inA,
			inB,
			out,
			mGridType,
			mOutside,
			h,
			padding,
			this->varBoolType()->currentKey());

		auto& phi = out.distances();
		uint ni = phi.nx();
		uint nj = phi.ny();
		uint nk = phi.nz();

		uint n_max_ijk = std::max(ni, std::max(nj, nk));

		// To ensure 2^N >= max(ni, nj, nk) / 2
		uint N = std::ceil(std::log2(0.5 * float(n_max_ijk)));

		DArray3D<GridType> gridtype_buffer(ni, nj, nk);
		DArray3D<Real> phi_buffer(ni, nj, nk);
		DArray3D<uint> alpha(ni, nj, nk);
		alpha.reset();

		for (int V = 0; V < V_max; V++) {
			if (V == 0) {
				for (int exp = 0; exp <= N; exp++) {
					uint interval = std::pow(2, exp);

					gridtype_buffer.assign(mGridType);
					phi_buffer.assign(phi);

					// x direction
					if (interval < ni)
					{
						cuExecute3D(make_uint3(ni - interval, nj, nk),
							CheckTentative_Boolean_X,
							phi,
							phi_buffer,
							mGridType,
							gridtype_buffer,
							alpha,
							mOutside,
							interval);

						LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
							phi,
							mGridType,
							alpha,
							mOutside,
							interval,
							h,
							true);
					}

					gridtype_buffer.assign(mGridType);

					// y direction
					if (interval < nj)
					{
						cuExecute3D(make_uint3(ni, nj - interval, nk),
							CheckTentative_Boolean_Y,
							phi,
							mGridType,
							gridtype_buffer,
							alpha,
							mOutside,
							interval);

						LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
							phi,
							mGridType,
							alpha,
							mOutside,
							interval,
							h,
							true);
					}

					gridtype_buffer.assign(mGridType);

					// z direction
					if (interval < nk)
					{
						cuExecute3D(make_uint3(ni, nj, nk - interval),
							CheckTentative_Boolean_Z,
							phi,
							mGridType,
							gridtype_buffer,
							alpha,
							mOutside,
							interval);

						LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
							phi,
							mGridType,
							alpha,
							mOutside,
							interval,
							h,
							true);
					}
				}

				for (int exp = N; exp >= 0; exp--) {
					uint interval = std::pow(2, exp);

					LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
						phi,
						mGridType,
						alpha,
						mOutside,
						interval,
						h,
						true);
				}
			}
			else {
				for (int exp = 0; exp <= N; exp++) {

					uint interval = std::pow(2, exp);

					LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
						phi,
						mGridType,
						alpha,
						mOutside,
						interval,
						h,
						true);
				}
				for (int exp = N; exp >= 0; exp--) {

					uint interval = std::pow(2, exp);

					LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
						phi,
						mGridType,
						alpha,
						mOutside,
						interval,
						h,
						true);
				}
			}
		}

		mGridType.clear();
		mOutside.clear();
		gridtype_buffer.clear();
		phi_buffer.clear();
		alpha.clear();
	}

	DEFINE_CLASS(MultiscaleFastIterativeMethodForBoolean);
}