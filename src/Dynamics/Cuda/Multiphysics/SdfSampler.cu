#include "SdfSampler.h"
#include "Topology/LevelSet.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "Matrix.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SdfSampler, TDataType)

	template<typename TDataType>
	SdfSampler<TDataType>::SdfSampler()
		: Sampler<TDataType>()
	{
		this->varSpacing()->setRange(0.02, 1);

		auto pts = std::make_shared<PointSet<TDataType>>();
		this->statePointSet()->setDataPtr(pts);

		this->statePointSet()->promoteOuput();
	}

	template<typename TDataType>
	SdfSampler<TDataType>::~SdfSampler()
	{

	}

	__global__ void C_PointCount(
		DArray3D<int> distance,
		DArray<int> VerticesFlag)
	{
		uint i = threadIdx.x + blockDim.x * blockIdx.x;
		uint j = threadIdx.y + blockDim.y * blockIdx.y;
		uint k = threadIdx.z + blockDim.z * blockIdx.z;

		uint nx = distance.nx();
		uint ny = distance.ny();
		uint nz = distance.nz();

		if (i >= nx || j >= ny || k >= nz) return;
		uint tId = i + j * nx + k * ny * nx;
		//hexahedronFlag[tId] = 0;

		if (i >= nx - 1 || j >= ny - 1 || k >= nz - 1) return;


		if (distance(i, j, k) == 1 &&
			distance(i + 1, j, k) == 1 &&

			distance(i + 1, j + 1, k) == 1 &&
			distance(i, j + 1, k) == 1 &&

			distance(i, j, k + 1) == 1 &&
			distance(i + 1, j, k + 1) == 1 &&

			distance(i + 1, j + 1, k + 1) == 1 &&
			distance(i, j + 1, k + 1) == 1) {

			uint index1 = distance.index(i, j, k);
			uint index2 = distance.index(i + 1, j, k);

			uint index3 = distance.index(i + 1, j + 1, k);
			uint index4 = distance.index(i, j + 1, k);

			uint index5 = distance.index(i, j, k + 1);
			uint index6 = distance.index(i + 1, j, k + 1);

			uint index7 = distance.index(i + 1, j + 1, k + 1);
			uint index8 = distance.index(i, j + 1, k + 1);

			atomicExch(&VerticesFlag[index1], 1);
			atomicExch(&VerticesFlag[index2], 1);
			atomicExch(&VerticesFlag[index3], 1);
			atomicExch(&VerticesFlag[index4], 1);
			atomicExch(&VerticesFlag[index5], 1);
			atomicExch(&VerticesFlag[index6], 1);
			atomicExch(&VerticesFlag[index7], 1);
			atomicExch(&VerticesFlag[index8], 1);
		}
	}

	template<typename Coord>
	__global__ void C_CalculatePoints(
		DArray<Coord> vertices,
		DArray<Coord> allVertices,
		DArray<int> VerticesFlag,
		DArray<int> VerticesFlagScan)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= allVertices.size()) return;

		if (VerticesFlag[pId] == 1) {
			vertices[VerticesFlagScan[pId] - 1] = allVertices[pId];
		}
	}

	template<typename TDataType, typename Real, typename Coord>
	__global__ void C_reconstructSDF(
		DistanceField3D<TDataType> inputSDF,
		DArray3D<int> distance,
		Coord minPoint,
		DArray<Coord> vertices,
		Real dx,
		DArray<int> VerticesFlag)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = distance.nx();
		uint ny = distance.ny();
		uint nz = distance.nz();

		if (i >= nx || j >= ny || k >= nz) return;
		VerticesFlag[i + j * nx + k * nx * ny] = 0;

		Coord point = minPoint + Coord(i * dx, j * dx, k * dx);

		Real a;
		Coord normal;
		inputSDF.getDistance(point, a, normal);

		vertices[i + j * nx + k * nx * ny] = point;
	}

	template<typename TDataType>
	void SdfSampler<TDataType>::resetStates()
	{
		Real dx = this->varSpacing()->getValue();

		auto vol = this->getVolume();
		auto& levelset = vol->stateLevelSet()->constDataPtr()->getSDF();

		Coord minPoint = levelset.lowerBound();
		Coord maxPoint = levelset.upperBound();

		uint ni = std::floor((maxPoint[0] - minPoint[0]) / dx) + 1;
		uint nj = std::floor((maxPoint[1] - minPoint[1]) / dx) + 1;
		uint nk = std::floor((maxPoint[2] - minPoint[2]) / dx) + 1;

		DArray3D<int> distance(ni + 1, nj + 1, nk + 1);
		DArray<Coord> allVertices((ni + 1) * (nj + 1) * (nk + 1));
		DArray<int> flags((ni + 1) * (nj + 1) * (nk + 1));

		cuExecute3D(make_uint3(distance.nx(), distance.ny(), distance.nz()),
			C_reconstructSDF,
			levelset,
			distance,
			minPoint,
			allVertices,
			dx,
			flags);

		cuExecute3D(make_uint3(distance.nx(), distance.ny(), distance.nz()),
			C_PointCount,
			distance,
			flags);


		CArray<int> CVerticesFlag;
		CVerticesFlag.assign(flags);

		CArray3D<int> Cdistance;
		Cdistance.assign(distance);

		Reduction<int> reduce;
		int verticesNum = reduce.accumulate(flags.begin(), flags.size());
		DArray<Coord> vertices;
		vertices.resize(verticesNum);
		DArray<int> VerticesFlagScan;
		VerticesFlagScan.assign(flags);
		thrust::inclusive_scan(thrust::device, VerticesFlagScan.begin(), VerticesFlagScan.begin() + VerticesFlagScan.size(), VerticesFlagScan.begin()); // in-place scan
		cuExecute(allVertices.size(),
			C_CalculatePoints,
			vertices,
			allVertices,
			flags,
			VerticesFlagScan);

		if (vertices.size() >= 0) {
			auto topo = this->statePointSet()->getDataPtr();
			topo->setPoints(vertices);
			topo->update();
		}

		distance.clear();
		allVertices.clear();
		flags.clear();
		CVerticesFlag.clear();
		Cdistance.clear();
		vertices.clear();
		VerticesFlagScan.clear();
	}

	template<typename TDataType>
	bool SdfSampler<TDataType>::validateInputs()
	{
		return this->getVolume() != nullptr;
	}

	DEFINE_CLASS(SdfSampler);
}