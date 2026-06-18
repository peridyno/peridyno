//get run time
#include "MultiscaleFastIterativeMethod.h"

#include "Algorithm/Reduction.h"

#include "Collision/Distance3D.h"
#include <cuda_runtime.h>

#include "Volume/BasicShapeToVolume.h"
#include "LevelSetConstructionAndBooleanHelper.h"

#include<cmath>
namespace dyno
{
	IMPLEMENT_TCLASS(MultiscaleFastIterativeMethod, TDataType)

		template<typename TDataType>
	MultiscaleFastIterativeMethod<TDataType>::MultiscaleFastIterativeMethod() : ComputeModule()
	{
	}

	template<typename TDataType>
	MultiscaleFastIterativeMethod<TDataType>::~MultiscaleFastIterativeMethod()
	{
	}

	template<typename TDataType>
	void MultiscaleFastIterativeMethod<TDataType>::compute()
	{
		auto tss = std::dynamic_pointer_cast<TriangleSets<TDataType>>(this->inTriangleSet()->constDataPtr());
		if (tss != nullptr)
		{
			this->makeLevelSetFromTriangleSets();
		}
		else
		{
			this->makeLevelSet();
		}
	}

	template<typename Real>
	__global__	void CheckTentative_MUTI_X(
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
			outside(i, j, k) = phiBuffer(i + interval, j, k) > 0 ? true : false;
 			phi(i, j, k) = phiBuffer(i + interval, j, k) > 0 ? REAL_MAX : -REAL_MAX;
		}

		if (type0 != GridType::Infinite && type1 == GridType::Infinite) {
			pointType(i + interval, j, k) = GridType::Tentative;
			alpha(i + interval, j, k) = interval;
			outside(i + interval, j, k) = phiBuffer(i, j, k) > 0 ? true : false;
 			phi(i + interval, j, k) = phiBuffer(i, j, k) > 0 ? REAL_MAX : -REAL_MAX;
		}
	}

	template<typename Real>
	__global__	void CheckTentative_MUTI_Y(
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
			outside(i, j, k) = phi(i, j + interval, k) > 0 ? true : false;
			phi(i, j, k) = phi(i, j + interval, k) > 0 ? REAL_MAX : -REAL_MAX;
		}

		if (type0 != GridType::Infinite && type1 == GridType::Infinite) {
			pointType(i, j + interval, k) = GridType::Tentative;
			alpha(i, j + interval, k) = interval;
			outside(i, j + interval, k) = phi(i, j, k) > 0 ? true : false;
			phi(i, j + interval, k) = phi(i, j, k) > 0 ? REAL_MAX : -REAL_MAX;
		}
	}

	template<typename Real>
	__global__	void CheckTentative_MUTI_Z(
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
			outside(i, j, k) = phi(i, j, k + interval) > 0 ? true : false;
			phi(i, j, k) = phi(i, j, k + interval) > 0 ? REAL_MAX : -REAL_MAX;
		}

		if (type0 != GridType::Infinite && type1 == GridType::Infinite) {
			pointType(i, j, k + interval) = GridType::Tentative;
			alpha(i, j, k + interval) = interval;
			outside(i, j, k + interval) = phi(i, j, k) > 0 ? true : false;
			phi(i, j, k + interval) = phi(i, j, k) > 0 ? REAL_MAX : -REAL_MAX;
		}
	}

	template<typename TDataType>
	void MultiscaleFastIterativeMethod<TDataType>::makeLevelSet()
	{
		if (this->outLevelSet()->isEmpty()) {
			this->outLevelSet()->allocate();
		}

		DistanceField3D<TDataType>& sdf = this->outLevelSet()->getDataPtr()->getSDF();

		Real h = this->varSpacing()->getValue();
		uint padding = this->varPadding()->getValue();
		uint V_max = this->varVCircle()->getValue();

		DArray3D<GridType> gridtype;
		DArray3D<int> closestTriId;

		Coord origin;
		LevelSetConstructionAndBooleanHelper<TDataType>::initialFromTriangle(
			this->inTriangleSet()->constDataPtr(),
			h,
			padding,
			sdf,
			origin,
			gridtype,
			closestTriId);
		closestTriId.clear();

		auto& phi = sdf.distances();
		uint ni = phi.nx();
		uint nj = phi.ny();
		uint nk = phi.nz();

		DArray3D<uint> alpha(ni, nj, nk);
		DArray3D<bool> outside(ni, nj, nk);
		alpha.reset();
		outside.reset();

		uint n_max_ijk = std::max(ni, std::max(nj, nk));

		// To ensure 2^N >= max(ni, nj, nk) / 2
		uint N = std::ceil(std::log2(0.5 * float(n_max_ijk)));

		DArray3D<GridType> gridtype_buffer(ni, nj, nk);
		DArray3D<Real> phi_buffer(ni, nj, nk);

		for (int V = 0; V < V_max; V++) {
			if (V == 0) {
				for (int exp = 0; exp <= N; exp++) {
					uint interval = std::pow(2, exp);

					gridtype_buffer.assign(gridtype);
					phi_buffer.assign(phi);

					// x direction
					if (interval < ni)
					{
						cuExecute3D(make_uint3(ni - interval, nj, nk),
							CheckTentative_MUTI_X,
							phi,
							phi_buffer,
							gridtype,
							gridtype_buffer,
							alpha,
							outside,
							interval);

						LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
							phi,
							gridtype,
							alpha,
							outside,
							interval,
							h,
							true);
					}

					gridtype_buffer.assign(gridtype);

					// y direction
					if (interval < nj)
					{
						cuExecute3D(make_uint3(ni, nj - interval, nk),
							CheckTentative_MUTI_Y,
							phi,
							gridtype,
							gridtype_buffer,
							alpha,
							outside,
							interval);

						LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
							phi,
							gridtype,
							alpha,
							outside,
							interval,
							h,
							true);
					}
	
					gridtype_buffer.assign(gridtype);

					// z direction
					if (interval < nk)
					{
						cuExecute3D(make_uint3(ni, nj, nk - interval),
							CheckTentative_MUTI_Z,
							phi,
							gridtype,
							gridtype_buffer,
							alpha,
							outside,
							interval);

						LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
							phi,
							gridtype,
							alpha,
							outside,
							interval,
							h,
							true);
					}
				}

				for (int exp = N; exp >= 0; exp--) {
					uint interval = std::pow(2, exp);

					LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
						phi,
						gridtype,
						alpha,
						outside,
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
						gridtype,
						alpha,
						outside,
						interval,
						h,
						true);
				}
				for (int exp = N; exp >= 0; exp--) {

					uint interval = std::pow(2, exp);

					LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
						phi,
						gridtype,
						alpha,
						outside,
						interval,
						h,
						true);
				}
			}
		}

		gridtype.clear();
		alpha.clear();
		outside.clear();
		gridtype_buffer.clear();
		phi_buffer.clear();
	}

	template<typename Real>
	__global__	void InitializeOutside(
		DArray3D<bool> outside,
		DArray3D<Real> distances,
		DArray3D<GridType> pointType)
	{
		int i = (blockIdx.x * blockDim.x + threadIdx.x);
		int j = (blockIdx.y * blockDim.y + threadIdx.y);
		int k = (blockIdx.z * blockDim.z + threadIdx.z);

		uint nx = pointType.nx();
		uint ny = pointType.ny();
		uint nz = pointType.nz();

		if (i >= nx || j >= ny || k >= nz) return;

		GridType type = pointType(i, j, k);

		if (type == GridType::Accepted || type == GridType::Tentative)
		{
			outside(i, j, k) = distances(i, j, k) >= 0 ? true : false;
		}
		else
		{
			//For Infinite points, temporarily set to true whose values will be fixed in later steps 
			outside(i, j, k) = true;
		}
	}

	__global__	void MFIM_CheckInside_MUTI_X(
		DArray3D<GridType> pointType,
		DArray3D<GridType> pointTypeBuffer,
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

		bool outside0 = outside(i, j, k);
		bool outside1 = outside(i + interval, j, k);

		if (type0 == GridType::Infinite && type1 != GridType::Infinite && outside1 == false) {
			pointType(i, j, k) = GridType::Tentative;
			outside(i, j, k) = false;
		}

		if (type0 != GridType::Infinite && type1 == GridType::Infinite && outside0 == false) {
			pointType(i + interval, j, k) = GridType::Tentative;
			outside(i + interval, j, k) = false;
		}
	}

	__global__	void MFIM_CheckInside_MUTI_Y(
		DArray3D<GridType> pointType,
		DArray3D<GridType> pointTypeBuffer,
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

		bool outside0 = outside(i, j, k);
		bool outside1 = outside(i, j + interval, k);

		if (type0 == GridType::Infinite && type1 != GridType::Infinite && outside1 == false) {
			pointType(i, j, k) = GridType::Tentative;
			outside(i, j, k) = false;
		}

		if (type0 != GridType::Infinite && type1 == GridType::Infinite && outside0 == false) {
			pointType(i, j + interval, k) = GridType::Tentative;
			outside(i, j + interval, k) = false;
		}
	}

	__global__	void MFIM_CheckInside_MUTI_Z(
		DArray3D<GridType> pointType,
		DArray3D<GridType> pointTypeBuffer,
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

		bool outside0 = outside(i, j, k);
		bool outside1 = outside(i, j, k + interval);

		if (type0 == GridType::Infinite && type1 != GridType::Infinite && outside1 == false) {
			pointType(i, j, k) = GridType::Tentative;
			outside(i, j, k) = false;
		}

		if (type0 != GridType::Infinite && type1 == GridType::Infinite && outside0 == false) {
			pointType(i, j, k + interval) = GridType::Tentative;
			outside(i, j, k + interval) = false;
		}
	}

	__global__	void CorrectFalseStatus(
		DArray3D<GridType> pointType,
		DArray3D<bool> outside)
	{
		int i = (blockIdx.x * blockDim.x + threadIdx.x);
		int j = (blockIdx.y * blockDim.y + threadIdx.y);
		int k = (blockIdx.z * blockDim.z + threadIdx.z);

		uint nx = pointType.nx();
		uint ny = pointType.ny();
		uint nz = pointType.nz();

		if (i >= nx - 1 || j >= ny - 1 || k >= nz - 1) return;
		if (i < 1 || j < 1 || k < 1) return;

		GridType type0 = pointType(i, j, k);

		if (type0 != GridType::Accepted)
			return;

		if ((pointType(i + 1, j, k) == GridType::Infinite ||
			pointType(i - 1, j, k) == GridType::Infinite ||
			pointType(i, j + 1, k) == GridType::Infinite ||
			pointType(i, j - 1, k) == GridType::Infinite ||
			pointType(i, j, k + 1) == GridType::Infinite ||
			pointType(i, j, k - 1) == GridType::Infinite) && outside(i, j, k) == true)
		{
			return;
		}

		if ((pointType(i + 1, j, k) == GridType::Tentative ||
			pointType(i - 1, j, k) == GridType::Tentative ||
			pointType(i, j + 1, k) == GridType::Tentative ||
			pointType(i, j - 1, k) == GridType::Tentative ||
			pointType(i, j, k + 1) == GridType::Tentative ||
			pointType(i, j, k - 1) == GridType::Tentative) && outside(i, j, k) == true)
		{
			pointType(i, j, k) = GridType::Tentative;
			outside(i, j, k) = false;
		}
	}

	template<typename Real>
	__global__	void ResetGridType(
		DArray3D<GridType> pointType,
		DArray3D<Real> phi,
		DArray3D<bool> outside)
	{
		int i = (blockIdx.x * blockDim.x + threadIdx.x);
		int j = (blockIdx.y * blockDim.y + threadIdx.y);
		int k = (blockIdx.z * blockDim.z + threadIdx.z);

		uint nx = pointType.nx();
		uint ny = pointType.ny();
		uint nz = pointType.nz();

		if (i >= nx || j >= ny || k >= nz) return;

		if (pointType(i, j, k) == GridType::Accepted && outside(i, j, k) == true)
		{
			pointType(i, j, k) = GridType::Accepted;
		}
		else
		{
			pointType(i, j, k) = GridType::Infinite;
			phi(i, j, k) = outside(i, j, k) ? REAL_MAX : -REAL_MAX;
		}
	}

	template<typename Real>
	__global__	void MFIM_CheckTentative_Boolean_X(
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
	__global__	void MFIM_CheckTentative_Boolean_Y(
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
	__global__	void MFIM_CheckTentative_Boolean_Z(
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
	void MultiscaleFastIterativeMethod<TDataType>::makeLevelSetFromTriangleSets()
	{
		if (this->outLevelSet()->isEmpty()) {
			this->outLevelSet()->allocate();
		}

		DistanceField3D<TDataType>& sdf = this->outLevelSet()->getDataPtr()->getSDF();

		Real h = this->varSpacing()->getValue();
		uint padding = this->varPadding()->getValue();
		uint V_max = this->varVCircle()->getValue();

		DArray3D<GridType> gridtype;
		DArray3D<int> closestTriId;

		auto tss = std::dynamic_pointer_cast<TriangleSets<TDataType>>(this->inTriangleSet()->constDataPtr());

		Coord origin;
		LevelSetConstructionAndBooleanHelper<TDataType>::initialFromTriangles(
			tss,
			h,
			padding,
			sdf,
			origin,
			gridtype,
			closestTriId);
		closestTriId.clear();

		auto& phi = sdf.distances();
		uint ni = phi.nx();
		uint nj = phi.ny();
		uint nk = phi.nz();

		DArray3D<uint> alpha(ni, nj, nk);
		DArray3D<bool> outside(ni, nj, nk);
		alpha.reset();
		outside.reset();

		uint n_max_ijk = std::max(ni, std::max(nj, nk));

		// To ensure 2^N >= max(ni, nj, nk) / 2
		uint N = std::ceil(std::log2(0.5 * float(n_max_ijk)));

		DArray3D<GridType> gridtype_buffer(ni, nj, nk);
		DArray3D<Real> phi_buffer(ni, nj, nk);

		cuExecute3D(make_uint3(ni, nj, nk),
			InitializeOutside,
			outside,
			phi,
			gridtype);

		//Identify Infinite points that are located inside the boundary and tag the corresponding values in the ouside array as false
		//TODO: reduce the iteration number
		for (int it = 0; it < n_max_ijk; it++)
		{
			gridtype_buffer.assign(gridtype);

			cuExecute3D(make_uint3(ni - 1, nj, nk),
				MFIM_CheckInside_MUTI_X,
				gridtype,
				gridtype_buffer,
				outside,
				1);

			gridtype_buffer.assign(gridtype);

			cuExecute3D(make_uint3(ni, nj - 1, nk),
				MFIM_CheckInside_MUTI_Y,
				gridtype,
				gridtype_buffer,
				outside,
				1);

			gridtype_buffer.assign(gridtype);

			cuExecute3D(make_uint3(ni, nj, nk - 1),
				MFIM_CheckInside_MUTI_Z,
				gridtype,
				gridtype_buffer,
				outside,
				1);
		}

		//Identify points that are located inside but are wrongly tagged as exterior points at the initialization stage. 
		for (uint it = 0; it < 5; it++)
		{
			cuExecute3D(make_uint3(ni, nj, nk),
				CorrectFalseStatus,
				gridtype,
				outside);
		}

		//Reset gridtype and phi for multiscale Eikonal equation solving
		cuExecute3D(make_uint3(ni, nj, nk),
			ResetGridType,
			gridtype,
			phi,
			outside);

		for (int V = 0; V < V_max; V++) {
			if (V == 0) {
				for (int exp = 0; exp <= N; exp++) {
					uint interval = std::pow(2, exp);

					gridtype_buffer.assign(gridtype);
					phi_buffer.assign(phi);

					// x direction
					if (interval < ni)
					{
						cuExecute3D(make_uint3(ni - interval, nj, nk),
							MFIM_CheckTentative_Boolean_X,
							phi,
							phi_buffer,
							gridtype,
							gridtype_buffer,
							alpha,
							outside,
							interval);

						LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
							phi,
							gridtype,
							alpha,
							outside,
							interval,
							h,
							true);
					}

					gridtype_buffer.assign(gridtype);

					// y direction
					if (interval < nj)
					{
						cuExecute3D(make_uint3(ni, nj - interval, nk),
							MFIM_CheckTentative_Boolean_Y,
							phi,
							gridtype,
							gridtype_buffer,
							alpha,
							outside,
							interval);

						LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
							phi,
							gridtype,
							alpha,
							outside,
							interval,
							h,
							true);
					}

					gridtype_buffer.assign(gridtype);

					// z direction
					if (interval < nk)
					{
						cuExecute3D(make_uint3(ni, nj, nk - interval),
							MFIM_CheckTentative_Boolean_Z,
							phi,
							gridtype,
							gridtype_buffer,
							alpha,
							outside,
							interval);

						LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
							phi,
							gridtype,
							alpha,
							outside,
							interval,
							h,
							true);
					}
				}

				for (int exp = N; exp >= 0; exp--) {
					uint interval = std::pow(2, exp);

					LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
						phi,
						gridtype,
						alpha,
						outside,
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
						gridtype,
						alpha,
						outside,
						interval,
						h,
						true);
				}
				for (int exp = N; exp >= 0; exp--) {

					uint interval = std::pow(2, exp);

					LevelSetConstructionAndBooleanHelper<TDataType>::fastIterative(
						phi,
						gridtype,
						alpha,
						outside,
						interval,
						h,
						true);
				}
			}
		}

		gridtype.clear();
		alpha.clear();
		outside.clear();
		gridtype_buffer.clear();
		phi_buffer.clear();
		closestTriId.clear();
	}

	DEFINE_CLASS(MultiscaleFastIterativeMethod);
}