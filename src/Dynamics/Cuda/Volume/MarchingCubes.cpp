#include "MarchingCubes.h"

#include "Module/MarchingCubesHelper.h"

#include "GLSurfaceVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	MarchingCubes<TDataType>::MarchingCubes()
		: Node()
	{
		this->varIsoValue()->setRange(-1.0f, 1.0f);
		this->varGridSpacing()->setRange(0.001, 1.0);

		auto renderer = std::make_shared<GLSurfaceVisualModule>();
		this->stateTriangleSet()->connect(renderer->inTriangleSet());
		this->graphicsPipeline()->pushModule(renderer);
	}

	template<typename TDataType>
	MarchingCubes<TDataType>::~MarchingCubes()
	{
	}


	template<typename TDataType>
	void MarchingCubes<TDataType>::constructSurfaceMesh()
	{

		auto sdfTopo = this->inLevelSet()->getDataPtr();
		auto isoValue = this->varIsoValue()->getValue();

		auto& sdf = sdfTopo->getSDF();
		
		Coord lowerBound = sdf.lowerBound();
		Coord upperBound = sdf.upperBound();

		Real h = this->varGridSpacing()->getValue();

		int nx = std::floor((upperBound[0] - lowerBound[0]) / h);
		int ny = std::floor((upperBound[1] - lowerBound[1]) / h);
		int nz = std::floor((upperBound[2] - lowerBound[2]) / h);

		DArray3D<Real> distances(nx, ny, nz);
		DArray<int> voxelVertNum((nx - 1) * (ny - 1) * (nz - 1));

		MarchingCubesHelper<TDataType>::reconstructSDF(
			distances,
			lowerBound,
			h,
			sdf);

		MarchingCubesHelper<TDataType>::countVerticeNumber(
			voxelVertNum,
			distances,
			isoValue,
			h);

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

		if (this->stateTriangleSet()->isEmpty()) {
			this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		}

		auto triSet = this->stateTriangleSet()->getDataPtr();
		triSet->setPoints(vertices);
		triSet->setTriangles(triangles);
		triSet->update();

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