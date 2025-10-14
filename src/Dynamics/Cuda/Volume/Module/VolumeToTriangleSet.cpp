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

		if (h < EPSILON)
			return false;

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
			iso,
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
			iso,
			h);

		if (this->outTriangleSet()->isEmpty()) {
			this->outTriangleSet()->allocate();
		}

		auto triSet = this->outTriangleSet()->getDataPtr();
		triSet->setPoints(vertices);
		triSet->setTriangles(triangles);
		triSet->update();

		distances.clear();
		voxelVertNum.clear();
		vertices.clear();
		triangles.clear();

		return true;
	}

	DEFINE_CLASS(VolumeToTriangleSet);
}