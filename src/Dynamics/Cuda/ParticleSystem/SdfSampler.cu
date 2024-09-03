#include "SdfSampler.h"
#include "Topology/SignedDistanceField.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "Matrix.h"
namespace dyno
{
	IMPLEMENT_TCLASS(SdfSampler, TDataType)

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

		if (i >= nx || j >= ny|| k >= nz) return;
		uint tId = i + j * nx  + k * ny  * nx ;
		//hexahedronFlag[tId] = 0;

		if (i >= nx - 1 || j >= ny - 1 || k >= nz -1) return;

		
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
		Vec3f cubeTilt,
		DArray<int> VerticesFlag,
		Mat3f cubeRotation,
		Mat3f Rotation)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		uint nx = distance.nx();
		uint ny = distance.ny();
		uint nz = distance.nz();

		if (i >= nx || j >= ny || k >= nz) return;
		distance(i, j, k) = 0;
		VerticesFlag[i + j * nx + k * nx * ny] = 0;

		float theta = (90.0f - cubeTilt[0]) / 180.0f * M_PI;
		float phi = (90.0f - cubeTilt[1]) / 180.0f * M_PI;
		float psi = (90.0f - cubeTilt[2]) / 180.0f * M_PI;


		Coord point = minPoint + Coord(i * dx, j * dx, k * dx);
	
		Real a;
		Coord normal;
		inputSDF.getDistance(point, a, normal);
	
		if (a <= 0) {
			atomicExch(&distance(i, j, k), 1);
		}
		


		point = minPoint + Coord(i * dx + (j * dx) / tan(theta),
			j * dx + (i * dx) / tan(phi), k * dx + (j * dx) / tan(psi));
		vertices[i + j * nx + k * nx * ny] = Rotation * cubeRotation * point;
	}

	template<typename TDataType>
	SdfSampler<TDataType>::SdfSampler()
		: Node()
	{
		this->varSpacing()->setRange(0.02, 1);
		this->varCubeTilt()->setRange(0, 80);
		this->varX()->setRange(0.001, 5);
		this->varY()->setRange(0.001, 5);
		this->varZ()->setRange(0.001, 5);

		this->varAlpha()->setRange(0, 360);
		this->varBeta()->setRange(0, 360);
		this->varGamma()->setRange(0, 360);
	}

	template<typename TDataType>
	SdfSampler<TDataType>::~SdfSampler()
	{

	}

	template<typename Real, typename Coord>
	__global__ void VS_SetupNodes(
		DArray<Coord> nodes,
		DArray3D<Real> distances,
		Coord origin,
		Real h)
	{
		uint i = threadIdx.x + blockDim.x * blockIdx.x;
		uint j = threadIdx.y + blockDim.y * blockIdx.y;
		uint k = threadIdx.z + blockDim.z * blockIdx.z;

		uint nx = distances.nx();
		uint ny = distances.ny();
		uint nz = distances.nz();

		if (i >= nx || j >= ny || k >= nz) return;

		Coord p = origin + h * Coord(i, j, k);
		nodes[distances.index(i, j, k)] = p;
	}


	template<typename Real>
	__global__ void VS_SetupSDFValues(
		DArray3D<Real> values,
		DArray<Real> distances)
	{
		uint i = threadIdx.x + blockDim.x * blockIdx.x;
		uint j = threadIdx.y + blockDim.y * blockIdx.y;
		uint k = threadIdx.z + blockDim.z * blockIdx.z;

		uint nx = values.nx();
		uint ny = values.ny();
		uint nz = values.nz();

		if (i >= nx || j >= ny || k >= nz) return;

		values(i, j, k) = distances[values.index(i, j, k)];
	}

	template<typename TDataType>
	std::shared_ptr<dyno::DistanceField3D<TDataType>> SdfSampler<TDataType>::convert2Uniform(
		VolumeOctree<TDataType>* volume,
		Real h)
	{
		Coord lo = volume->lowerBound();
		Coord hi = volume->upperBound();

		int nx = (hi[0] - lo[0]) / h;
		int ny = (hi[1] - lo[1]) / h;
		int nz = (hi[2] - lo[2]) / h;

		auto levelset = std::make_shared<DistanceField3D<TDataType>>();
		levelset->setSpace(lo, hi, nx, ny, nz);

		auto& distances = levelset->getMDistance();

		DArray<Coord> nodes(distances.size());

		cuExecute3D(make_uint3(distances.nx(), distances.ny(), distances.nz()),
			VS_SetupNodes,
			nodes,
			distances,
			lo,
			h);

		DArray<Real> sdfValues;
		DArray<Coord> normals;

		volume->stateSDFTopology()->constDataPtr()->getSignDistance(nodes, sdfValues, normals);

		cuExecute3D(make_uint3(distances.nx(), distances.ny(), distances.nz()),
			VS_SetupSDFValues,
			distances,
			sdfValues);

		nodes.clear();
		sdfValues.clear();
		normals.clear();

		return levelset;
	}

	template<typename TDataType>
	void SdfSampler<TDataType>::resetStates()
	{
		auto vol = this->getVolume();
		Real dx = this->varSpacing()->getData();

		//Adaptive mesh to uniform mesh
		auto inputSDF = this->convert2Uniform(vol, dx);


		if (this->outPointSet()->isEmpty()) {
			auto pts = std::make_shared<PointSet<TDataType>>();
			this->outPointSet()->setDataPtr(pts);
		}

		Coord minPoint = inputSDF->lowerBound();
		Coord maxPoint = inputSDF->upperBound();

		Vec3f cubeTilt = this->varCubeTilt()->getData();
		Vec3f Xax = this->varX()->getData();
		Vec3f Yax = this->varY()->getData();
		Vec3f Zax = this->varZ()->getData();
		Mat3f cubeRotation(Xax, Yax, Zax);
		cubeRotation = cubeRotation.transpose();

		uint ni = std::floor((maxPoint[0] - minPoint[0]) / dx) + 1;
		uint nj = std::floor((maxPoint[1] - minPoint[1]) / dx) + 1;
		uint nk = std::floor((maxPoint[2] - minPoint[2]) / dx) + 1;

		DArray3D<int> distance(ni + 1, nj + 1, nk + 1);
		DArray<Coord> allVertices((ni + 1) * (nj + 1) * (nk + 1));
		DArray<int> VerticesFlag((ni + 1) * (nj + 1) * (nk + 1));

		float Alpha = this->varAlpha()->getData() / 180.0f * M_PI;
		float Beta = this->varBeta()->getData() / 180.0f * M_PI;
		float Gamma = this->varGamma()->getData() / 180.0f * M_PI;
		Mat3f Rotation(cos(Alpha) * cos(Beta), cos(Alpha) * sin(Beta) * sin(Gamma) - sin(Alpha) * cos(Gamma), cos(Alpha) * sin(Beta) * cos(Gamma) + sin(Alpha) * sin(Gamma),
			sin(Alpha) * cos(Beta), sin(Alpha) * sin(Beta) * sin(Gamma) + cos(Alpha) * cos(Gamma), sin(Alpha) * sin(Beta) * cos(Gamma) - cos(Alpha) * sin(Gamma),
			-sin(Beta), cos(Beta) * sin(Gamma), cos(Beta) * cos(Gamma));

		cuExecute3D(make_uint3(distance.nx(), distance.ny(), distance.nz()),
			C_reconstructSDF,
			*inputSDF,
			distance,
			minPoint,
			allVertices,
			dx,
			cubeTilt,
			VerticesFlag,
			cubeRotation,
			Rotation);

		cuExecute3D(make_uint3(distance.nx(), distance.ny(), distance.nz()),
			C_PointCount,
			distance,
			VerticesFlag);
		

		CArray<int> CVerticesFlag;
		CVerticesFlag.assign(VerticesFlag);

		CArray3D<int> Cdistance;
		Cdistance.assign(distance);

		Reduction<int> reduce;
		int verticesNum = reduce.accumulate(VerticesFlag.begin(), VerticesFlag.size());
		DArray<Coord> vertices;
		vertices.resize(verticesNum);
		DArray<int> VerticesFlagScan;
		VerticesFlagScan.assign(VerticesFlag);
		thrust::inclusive_scan(thrust::device, VerticesFlagScan.begin(), VerticesFlagScan.begin() + VerticesFlagScan.size(), VerticesFlagScan.begin()); // in-place scan
		cuExecute(allVertices.size(),
			C_CalculatePoints,
			vertices,
			allVertices,
			VerticesFlag,
			VerticesFlagScan);

		if (vertices.size() >= 0) {
			auto topo = this->outPointSet()->getDataPtr();
			topo->setPoints(vertices);
			topo->update();
		}
	}

	template<typename TDataType>
	bool dyno::SdfSampler<TDataType>::validateInputs()
	{
		return this->getVolume() != nullptr;
	}

	DEFINE_CLASS(SdfSampler);
}