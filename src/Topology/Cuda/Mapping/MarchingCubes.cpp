#include "MarchingCubes.h"

#include "MarchingCubesHelper.h"

namespace dyno
{
	template<typename TDataType>
	MarchingCubes<TDataType>::MarchingCubes()
		: Node()
	{
		this->varGridSpacing()->setRange(0.001, 1.0);
	}

	template<typename TDataType>
	MarchingCubes<TDataType>::~MarchingCubes()
	{
	}


	template<typename TDataType>
	void MarchingCubes<TDataType>::constructSurfaceMesh()
	{

		auto sdfTopo = this->inLevelSet()->getDataPtr();
		auto isoValue = this->varIsoValue()->getData();

		auto& sdf = sdfTopo->getSDF();
		
		Coord lowerBound = sdf.lowerBound();
		Coord upperBound = sdf.upperBound();

		Real h = this->varGridSpacing()->getData();

		int nx = (upperBound[0] - lowerBound[0]) / h;
		int ny = (upperBound[1] - lowerBound[1]) / h;
		int nz = (upperBound[2] - lowerBound[2]) / h;

		DArray3D<Real> distances(nx + 1, ny + 1, nz + 1);
		DArray<int> voxelVertNum(nx * ny * nz);

		MarchingCubesHelper<TDataType>::reconstructSDF(
			distances,
			lowerBound,
			h,
			sdf);

		MarchingCubesHelper<TDataType>::countVerticeNumber(
			voxelVertNum,
			distances,
			isoValue);

		Reduction<int> reduce;
		int totalVNum = reduce.accumulate(voxelVertNum.begin(), voxelVertNum.size());

		Scan<int> scan;
		scan.exclusive(voxelVertNum.begin(), voxelVertNum.size());

		DArray<Coord> vertices(totalVNum);

		DArray<TopologyModule::Triangle> triangles(totalVNum / 3);

		MarchingCubesHelper<TDataType>::constructTriangles(
			vertices,
			triangles,
			voxelVertNum,
			distances,
			lowerBound,
			isoValue,
			h);

		if (this->outTriangleSet()->isEmpty()) {
			this->outTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		}

		auto triSet = this->outTriangleSet()->getDataPtr();
		triSet->setPoints(vertices);
		triSet->setTriangles(triangles);

		distances.clear();
		voxelVertNum.clear();
		vertices.clear();
		triangles.clear();
	}


	template<typename TDataType>
	void MarchingCubes<TDataType>::resetStates()
	{
		this->constructSurfaceMesh();
	}


	template<typename TDataType>
	void MarchingCubes<TDataType>::updateStates()
	{
		this->constructSurfaceMesh();
	}

	DEFINE_CLASS(MarchingCubes);
}