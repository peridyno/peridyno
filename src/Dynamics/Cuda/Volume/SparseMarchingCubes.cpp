#include "SparseMarchingCubes.h"

#include "Mapping/MarchingCubesHelper.h"

#include "GLSurfaceVisualModule.h"

#include "VolumeHelper.h"

namespace dyno
{
	template<typename TDataType>
	SparseMarchingCubes<TDataType>::SparseMarchingCubes()
		: Node()
	{
		auto module = std::make_shared<GLSurfaceVisualModule>();
		this->stateTriangleSet()->connect(module->inTriangleSet());
		this->graphicsPipeline()->pushModule(module);
	}

	template<typename TDataType>
	SparseMarchingCubes<TDataType>::~SparseMarchingCubes()
	{
	}

	template<typename TDataType>
	void SparseMarchingCubes<TDataType>::resetStates()
	{
		auto sv = this->getSparseVolume();

		if (sv->stateSDFTopology()->isEmpty())
		{
			printf("SparseMarchingCubes: The import is empty! \n");
			return;
		}

		Real isoValue = this->varIsoValue()->getData();

		//auto sv = this->getSparseVolume();


		auto octree = sv->stateSDFTopology()->getDataPtr();

		DArray<Coord> ceilVertices;

		octree->getCellVertices0(ceilVertices);

		DArray<Real> sdfs;
		DArray<Coord> normals;

		octree->getSignDistanceMLS(ceilVertices, sdfs, normals, false);
		//sv->getSignDistanceKernel(ceilVertices, sdfs);

		//DArray3D<Real> distances(nx + 1, ny + 1, nz + 1);
		DArray<uint> voxelVertNum(ceilVertices.size() / 8);

		MarchingCubesHelper<TDataType>::countVerticeNumberForOctree(
			voxelVertNum,
			ceilVertices,
			sdfs,
			isoValue);

		Reduction<uint> reduce;
		uint totalVNum = reduce.accumulate(voxelVertNum.begin(), voxelVertNum.size());

		Scan<uint> scan;
		scan.exclusive(voxelVertNum.begin(), voxelVertNum.size());

		DArray<Coord> triangleVertices(totalVNum);

		DArray<TopologyModule::Triangle> triangles(totalVNum / 3);

		MarchingCubesHelper<TDataType>::constructTrianglesForOctree(
			triangleVertices,
			triangles,
			voxelVertNum,
			ceilVertices,
			sdfs,
			isoValue);

		if (this->stateTriangleSet()->isEmpty()) {
			this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		}

		auto triSet = this->stateTriangleSet()->getDataPtr();
		triSet->setPoints(triangleVertices);
		triSet->setTriangles(triangles);

		sdfs.clear();
		normals.clear();
		voxelVertNum.clear();
		ceilVertices.clear();
		triangleVertices.clear();
		triangles.clear();
	}

	template<typename TDataType>
	void SparseMarchingCubes<TDataType>::updateStates()
	{
		this->reset();
	}

	template<typename TDataType>
	bool SparseMarchingCubes<TDataType>::validateInputs()
	{
		return this->getSparseVolume() != nullptr;
	}

	DEFINE_CLASS(SparseMarchingCubes);
}