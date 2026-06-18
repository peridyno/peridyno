#include "VolumeToTriangleSet.h"

#include "MarchingCubesHelper.h"

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

		auto levelset = this->inVolume()->constDataPtr();

		auto& sdf = levelset->getSDF();

		Coord lowerBound = sdf.lowerBound();
		Coord upperBound = sdf.upperBound();

		Real h = sdf.getGridSpacing();

		auto distances = sdf.distances();

		int nx = distances.nx() - 1;
		int ny = distances.ny() - 1;
		int nz = distances.nz() - 1;

		if (nx <= 0 || ny <= 0 || nz <= 0)
			return false;

		if (h < EPSILON)
			return false;

//		DArray3D<Real> distances(nx + 1, ny + 1, nz + 1);
		DArray<int> voxelVertNum(nx * ny * nz);

// 		MarchingCubesHelper<TDataType>::reconstructSDF(
// 			distances,
// 			lowerBound,
// 			h,
// 			sdf);

		MarchingCubesHelper<TDataType>::countVerticeNumber(
			voxelVertNum,
			distances,
			iso,
			h);

		Reduction<int> reduce;
		int totalVNum = reduce.accumulate(voxelVertNum.begin(), voxelVertNum.size());

		Scan<int> scan;
		scan.exclusive(voxelVertNum.begin(), voxelVertNum.size());

		DArray<Coord> vertices(totalVNum);

		DArray<Topology::Triangle> triangles(totalVNum / 3);

		MarchingCubesHelper<TDataType>::constructTriangles(
			vertices,
			triangles,
			voxelVertNum,
			distances,
			lowerBound,
			iso,
			h);

		if (this->outTriangleSet()->isEmpty()) {
			this->outTriangleSet()->allocate();
		}

		auto triSet = this->outTriangleSet()->getDataPtr();
		triSet->setPoints(vertices);
		triSet->setTriangles(triangles);
		triSet->update();

		voxelVertNum.clear();
		vertices.clear();
		triangles.clear();

		return true;
	}

	DEFINE_CLASS(VolumeToTriangleSet);
}