#include "FastMarchingMethodGPU.h"

#include "Algorithm/Reduction.h"

#include "Collision/Distance3D.h"

#include "MarchingCubesHelper.h"

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

	__device__ void FSM_SWAP(
		Real& a,
		Real& b)
	{
		Real tmp = b;
		b = a;
		a = tmp;
	}

	//Refer to Algorithm 1.3 in "Improved Fast Iterative Algorithm for Eikonal Equation for GPU Computing", Yuhao Huang 2021
	__device__ void UpdatePhi(
		DArray3D<Real>& phi,
		DArray3D<GridType>& type,
		DArray3D<bool>& outside,
		uint i, uint j, uint k,
		Real dx)
	{
		bool outside_ijk = outside(i, j, k);
	
		uint nx = phi.nx();
		uint ny = phi.ny();
		uint nz = phi.nz();

		Real phi_minx, phi_miny, phi_minz;

		if (i == 0) phi_minx = phi(1, j, k);
		else if (i == nx - 1) phi_minx = phi(nx - 2, j, k);
		else phi_minx = outside_ijk ? minimum(phi(i - 1, j, k), phi(i + 1, j, k)) : maximum(phi(i - 1, j, k), phi(i + 1, j, k));

		if (j == 0) phi_miny = phi(i, 1, k);
		else if (j == ny - 1) phi_miny = phi(i, ny - 2, k);
		else phi_miny = outside_ijk ? minimum(phi(i, j - 1, k), phi(i, j + 1, k)) : maximum(phi(i, j - 1, k), phi(i, j + 1, k));

		if (k == 0) phi_minz = phi(i, j, 1);
		else if (k == nz - 1) phi_minz = phi(i, j, nz - 2);
		else phi_minz = outside_ijk ? minimum(phi(i, j, k - 1), phi(i, j, k + 1)) : maximum(phi(i, j, k - 1), phi(i, j, k + 1));

		Real a[3];
		a[0] = phi_minx;
		a[1] = phi_miny;
		a[2] = phi_minz;

		// Sort
		if (outside_ijk)
		{
			//Ascending
			if (a[0] > a[1]) FSM_SWAP(a[0], a[1]);
			if (a[1] > a[2]) FSM_SWAP(a[1], a[2]);
			if (a[0] > a[1]) FSM_SWAP(a[0], a[1]);
		}
		else
		{
			//Descending
			if (a[0] < a[1]) FSM_SWAP(a[0], a[1]);
			if (a[1] < a[2]) FSM_SWAP(a[1], a[2]);
			if (a[0] < a[1]) FSM_SWAP(a[0], a[1]);
		}

		Real phi_ijk;
		Real sum_a = a[0] + a[1] + a[2];
		Real sum_a2 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];

		Real sign = outside_ijk ? Real(1) : Real(-1);

		if (glm::abs(a[0] - a[2]) < dx)
		{
			phi_ijk = (2 * sum_a + sign * glm::sqrt(4 * sum_a * sum_a - 12 * (sum_a2 - dx * dx))) / Real(6);
		}
		else if (glm::abs(a[0] - a[1]) < dx)
		{
			phi_ijk = 0.5 * (a[0] + a[1] + sign * glm::sqrt(2 * dx * dx - (a[0] - a[1]) * (a[0] - a[1])));
		}
		else
		{
			phi_ijk = a[0] + sign * dx;
		}

		Real phi_ijk_old = phi(i, j, k);

		phi(i, j, k) = phi_ijk;


		if (glm::abs(phi_ijk_old - phi_ijk) < EPSILON)
		{
			type(i, j, k) = GridType::Accepted;
		}
	}

	template<typename Real>
	__global__	void FSMI_FastMarching(
		DArray3D<Real> phi,
		DArray3D<GridType> pointType,
		DArray3D<bool> outside,
		Real dx)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = phi.nx();
		uint ny = phi.ny();
		uint nz = phi.nz();

		if (i > nx || j >= ny || k >= nz) return;

		if (pointType(i, j, k) != GridType::Tentative)
			return;

		UpdatePhi(phi, pointType, outside, i, j, k, dx);
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

		if (type0 == GridType::Accepted && type1 != GridType::Accepted)
			pointType(i + 1, j, k) = GridType::Tentative;

		if (type0 != GridType::Accepted && type1 == GridType::Accepted)
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

		if (type0 == GridType::Accepted && type1 != GridType::Accepted)
			pointType(i, j + 1, k) = GridType::Tentative;

		if (type0 != GridType::Accepted && type1 == GridType::Accepted)
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

		if (type0 == GridType::Accepted && type1 != GridType::Accepted)
			pointType(i, j, k + 1) = GridType::Tentative;

		if (type0 != GridType::Accepted && type1 == GridType::Accepted)
			pointType(i, j, k) = GridType::Tentative;
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void FSMI_BooleanOp(
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
		fieldA.getDistance(point, a);

		Real b;
		fieldB.getDistance(point, b);

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
			outside(i, j, k) = (Outside(a, iso, dx) && Outside(b, iso, dx) && op < 3.5 * dx) ? true : false;
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
	void FastMarchingMethodGPU<TDataType>::compute()
	{
		auto& inA = this->inLevelSetA()->getDataPtr()->getSDF();
		auto& inB = this->inLevelSetB()->getDataPtr()->getSDF();

		if (this->outLevelSet()->isEmpty()) {
			this->outLevelSet()->allocate();
		}

		DistanceField3D<TDataType>& out = this->outLevelSet()->getDataPtr()->getSDF();

		//Calculate the bounding box
		Coord min_box(std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max(), std::numeric_limits<Real>::max());
		Coord max_box(-std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max(), -std::numeric_limits<Real>::max());

		min_box = min_box.minimum(inA.lowerBound());
		min_box = min_box.minimum(inB.lowerBound());

		max_box = max_box.maximum(inA.upperBound());
		max_box = max_box.maximum(inB.upperBound());

		//Align the bounding box with a background grid to avoid flickering artifacts
		Real dx = this->varSpacing()->getValue();

		int min_i = std::floor(min_box[0] / dx);
		int min_j = std::floor(min_box[1] / dx);
		int min_k = std::floor(min_box[2] / dx);

		int max_i = std::ceil(max_box[0] / dx);
		int max_j = std::ceil(max_box[1] / dx);
		int max_k = std::ceil(max_box[2] / dx);

		int ni = max_i - min_i + 1;
		int nj = max_j - min_j + 1;
		int nk = max_k - min_k + 1;

		min_box = Coord(min_i * dx, min_j * dx, min_k * dx);
		max_box = Coord(max_i * dx, max_j * dx, max_k * dx);

		out.setSpace(min_box, max_box, dx);

		auto& phi = out.distances();

		if (ni != mGridType.nx() || nj != mGridType.ny() || nk != mGridType.nz()) {
			mGridType.resize(ni, nj, nk);
			mOutside.resize(ni, nj, nk);
		}

		//Calculate the boolean of two distance fields
		cuExecute3D(make_uint3(ni, nj, nk),
			FSMI_BooleanOp,
			phi,
			mGridType,
			mOutside,
			inA,
			inB,
			out.lowerBound(),
			out.getGridSpacing(),
			this->varBoolType()->currentKey());

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

			cuExecute3D(make_uint3(ni, nj, nk),
				FSMI_FastMarching,
				phi,
				mGridType,
				mOutside,
				dx);
		}

		mGridType.clear();
		mOutside.clear();
	}

	DEFINE_CLASS(FastMarchingMethodGPU);
}