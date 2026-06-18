#include <fstream>
#include "DistanceField3D.h"
#include "Vector.h"
#include "DataTypes.h"

namespace dyno
{
	template<typename TDataType>
	DistanceField3D<TDataType>::DistanceField3D()
	{
	}

	template<typename TDataType>
	DistanceField3D<TDataType>::DistanceField3D(std::string filename)
	{
		loadSDF(filename);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::setSpace(const Coord p0, const Coord p1, Real h)
	{
		mOrigin = p0;
		mH = h;

		uint nbx = std::ceil((p1.x - p0.x) / h);
		uint nby = std::ceil((p1.y - p0.y) / h);
		uint nbz = std::ceil((p1.z - p0.z) / h);

		mDistances.resize(nbx+1, nby+1, nbz+1);
	}

	template<typename TDataType>
	DistanceField3D<TDataType>::~DistanceField3D()
	{
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::translate(const Coord &t) {
		mOrigin += t;
	}

	template <typename Real>
	__global__ void K_Scale(DArray3D<Real> distance, float s)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.nx()) return;
		if (j >= distance.ny()) return;
		if (k >= distance.nz()) return;

		distance(i, j, k) = s*distance(i, j, k);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::scale(const Real s) {
		mOrigin[0] *= s;
		mOrigin[1] *= s;
		mOrigin[2] *= s;
		mH *= s;

		dim3 blockSize = make_uint3(8, 8, 8);
		dim3 gridDims = cudaGridSize3D(make_uint3(mDistances.nx(), mDistances.ny(), mDistances.nz()), blockSize);

		K_Scale << <gridDims, blockSize >> >(mDistances, s);
	}

	template<typename Real>
	__global__ void K_Invert(DArray3D<Real> distance)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.nx()) return;
		if (j >= distance.ny()) return;
		if (k >= distance.nz()) return;

		distance(i, j, k) = -distance(i, j, k);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::invertSDF()
	{
		dim3 blockSize = make_uint3(8, 8, 8);
		dim3 gridDims = cudaGridSize3D(make_uint3(mDistances.nx(), mDistances.ny(), mDistances.nz()), blockSize);

		K_Invert << <gridDims, blockSize >> >(mDistances);
	}

	template <typename Real, typename Coord>
	__global__ void D3D_InitializeFromBox(
		DArray3D<Real> distance,
		TOrientedBox3D<Real> obb,
		Coord start, 
		Real h, 
		bool inverted)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.nx()) return;
		if (j >= distance.ny()) return;
		if (k >= distance.nz()) return;

		Real sign = inverted ? -1.0f : 1.0f;

		TPoint3D<Real> p(start + Coord(i, j, k) * h);

		distance(i, j, k) = sign*p.distance(obb);
	}

	template<typename TDataType>
	void dyno::DistanceField3D<TDataType>::loadAABB(Coord lo, Coord hi, bool inverted /*= false*/)
	{
		mInverted = inverted;

		TOrientedBox3D<Real> obb;
		obb.center = 0.5 * (lo + hi);
		obb.extent = 0.5 * (hi - lo);

		cuExecute3D(make_uint3(mDistances.nx(), mDistances.ny(), mDistances.nz()),
			D3D_InitializeFromBox,
			mDistances,
			obb,
			mOrigin,
			mH,
			inverted);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::loadBox(TOrientedBox3D<Real> obb, bool inverted)
	{
		mInverted = inverted;

		cuExecute3D(make_uint3(mDistances.nx(), mDistances.ny(), mDistances.nz()),
			D3D_InitializeFromBox,
			mDistances,
			obb,
			mOrigin,
			mH,
			inverted);
	}

	template <typename Real, typename Coord>
	__global__ void D3D_InitializeFromCylinder(
		DArray3D<Real> distance, 
		TCylinder3D<Real> cylinder,
		Coord start, 
		Real h, 
		bool inverted)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.nx()) return;
		if (j >= distance.ny()) return;
		if (k >= distance.nz()) return;

		Real sign = inverted ? -1.0f : 1.0f;

		TPoint3D<Real> p(start + Coord(i, j, k) * h);

		distance(i, j, k) = sign* p.distance(cylinder);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::loadCylinder(TCylinder3D<Real> cylinder, bool inverted)
	{
		mInverted = inverted;

		cuExecute3D(make_uint3(mDistances.nx(), mDistances.ny(), mDistances.nz()),
			D3D_InitializeFromCylinder,
			mDistances,
			cylinder,
			mOrigin,
			mH,
			inverted);
	}

	template <typename Real, typename Coord>
	__global__ void D3D_InitializeFromSphere(
		DArray3D<Real> distance, 
		TSphere3D<Real> sphere,
		Coord start, 
		Real h, 
		bool inverted)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.nx()) return;
		if (j >= distance.ny()) return;
		if (k >= distance.nz()) return;

		Real sign = inverted ? -1.0f : 1.0f;

		TPoint3D<Real> p(start + Coord(i, j, k) * h);

		distance(i, j, k) = sign * p.distance(sphere);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::loadSphere(TSphere3D<Real> sphere, bool inverted)
	{
		mInverted = inverted;

		cuExecute3D(make_uint3(mDistances.nx(), mDistances.ny(), mDistances.nz()),
			D3D_InitializeFromSphere,
			mDistances, 
			sphere, 
			mOrigin, 
			mH, 
			inverted);
	}

	template <typename Real, typename Coord>
	__global__ void D3D_InitializeFromCapsule(
		DArray3D<Real> distance, 
		TCapsule3D<Real> capsule,
		Coord start, 
		Real h,
		bool inverted)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.nx()) return;
		if (j >= distance.ny()) return;
		if (k >= distance.nz()) return;

		Real sign = inverted ? -1.0f : 1.0f;

		TPoint3D<Real> p(start + Coord(i, j, k) * h);

		distance(i, j, k) = sign * p.distance(capsule);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::loadCapsule(TCapsule3D<Real> capsule, bool inverted /*= false*/)
	{
		mInverted = inverted;

		cuExecute3D(make_uint3(mDistances.nx(), mDistances.ny(), mDistances.nz()),
			D3D_InitializeFromCapsule,
			mDistances,
			capsule,
			mOrigin, 
			mH,
			inverted);
	}

	template <typename Real, typename Coord>
	__global__ void D3D_InitializeFromCone(
		DArray3D<Real> distance,
		TCone3D<Real> cone,
		Coord start,
		Real h,
		bool inverted)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.nx()) return;
		if (j >= distance.ny()) return;
		if (k >= distance.nz()) return;

		Real sign = inverted ? -1.0f : 1.0f;

		TPoint3D<Real> p(start + Coord(i, j, k) * h);

		distance(i, j, k) = sign * p.distance(cone);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::loadCone(TCone3D<Real> cone, bool inverted /*= false*/)
	{
		mInverted = inverted;

		cuExecute3D(make_uint3(mDistances.nx(), mDistances.ny(), mDistances.nz()),
			D3D_InitializeFromCone,
			mDistances,
			cone,
			mOrigin,
			mH,
			inverted);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::loadSDF(std::string filename, bool inverted)
	{
		std::ifstream input(filename.c_str(), std::ios::in);
		if (!input.is_open())
		{
			std::cout << "Reading file " << filename << " error!" << std::endl;
			return;
		}

		int nbx, nby, nbz;
		int xx, yy, zz;

		input >> xx;
		input >> yy;
		input >> zz;

		input >> mOrigin[0];
		input >> mOrigin[1];
		input >> mOrigin[2];

		Real t_h;
		input >> t_h;

		std::cout << "SDF: " << xx << ", " << yy << ", " << zz << std::endl;
		std::cout << "SDF: " << mOrigin[0] << ", " << mOrigin[1] << ", " << mOrigin[2] << std::endl;
		std::cout << "SDF: " << mOrigin[0] + t_h*xx << ", " << mOrigin[1] + t_h*yy << ", " << mOrigin[2] + t_h*zz << std::endl;

		nbx = xx;
		nby = yy;
		nbz = zz;
		mH = t_h;

		CArray3D<Real> distances(nbx, nby, nbz);
		for (int k = 0; k < zz; k++) {
			for (int j = 0; j < yy; j++) {
				for (int i = 0; i < xx; i++) {
					float dist;
					input >> dist;
					distances(i, j, k) = dist;
				}
			}
		}
		input.close();

		mDistances.resize(nbx, nby, nbz);
		mDistances.assign(distances);

		mInverted = inverted;
		if (inverted)
		{
			invertSDF();
		}

		std::cout << "read data successful" << std::endl;
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::release()
	{
		mDistances.clear();
	}

	template class DistanceField3D<DataType3f>;
	template class DistanceField3D<DataType3d>;
}