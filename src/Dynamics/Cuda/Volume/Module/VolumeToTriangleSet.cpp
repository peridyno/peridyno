#include "VolumeToTriangleSet.h"

#include "Mapping/MarchingCubesHelper.h"

namespace dyno
{
	template<typename TDataType>
	VolumeToTriangleSet<TDataType>::VolumeToTriangleSet()
		: TopologyMapping()
	{

	}

	template<typename TDataType>
	VolumeToTriangleSet<TDataType>::~VolumeToTriangleSet()
	{

	}

	template<typename TDataType>
	bool VolumeToTriangleSet<TDataType>::apply()
	{
		Real iso = this->varIsoValue()->getValue();

		auto vol = this->ioVolume()->constDataPtr();

		DArray<Coord> ceilVertices;

		vol->getCellVertices0(ceilVertices);

		DArray<Real> sdfs;
		DArray<Coord> normals;

		vol->getSignDistanceMLS(ceilVertices, sdfs, normals, false);
		//sv->getSignDistanceKernel(ceilVertices, sdfs);

		//DArray3D<Real> distances(nx + 1, ny + 1, nz + 1);
		DArray<uint> voxelVertNum(ceilVertices.size() / 8);

		MarchingCubesHelper<TDataType>::countVerticeNumberForOctree(
			voxelVertNum,
			ceilVertices,
			sdfs,
			iso);

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
			iso);

		if (this->outTriangleSet()->isEmpty()) {
			this->outTriangleSet()->allocate();
		}

		auto triSet = this->outTriangleSet()->getDataPtr();
		triSet->setPoints(triangleVertices);
		triSet->setTriangles(triangles);
		triSet->update();

		sdfs.clear();
		normals.clear();
		voxelVertNum.clear();
		ceilVertices.clear();
		triangleVertices.clear();
		triangles.clear();

		return true;
	}

	DEFINE_CLASS(VolumeToTriangleSet);
}