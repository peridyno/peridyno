#include "VolumeClipper.h"

#include "MarchingCubesHelper.h"

namespace dyno
{
	template<typename TDataType>
	VolumeClipper<TDataType>::VolumeClipper()
		: Node()
	{
		auto levelSet = std::make_shared<SignedDistanceField<TDataType>>();
		this->inLevelSet()->setDataPtr(levelSet);
	}

	template<typename TDataType>
	VolumeClipper<TDataType>::~VolumeClipper()
	{
	}

	template<typename TDataType>
	void VolumeClipper<TDataType>::resetStates()
	{
		auto center = this->varTranslation()->getData();
		auto eulerAngles = this->varRotation()->getData();

		auto levelSet = this->inLevelSet()->getDataPtr()->getSDF();
		levelSet.loadSDF(getAssetPath() + "bowl/bowl.sdf", false);

		Quat<Real> q = Quat<Real>::fromEulerAngles(eulerAngles[0], eulerAngles[1], eulerAngles[2]);

		int nx = levelSet.nx();
		int ny = levelSet.ny();
		int nz = levelSet.nz();

		DArray<int> voxelVertNum(nx * ny * nz);

		MarchingCubesHelper<TDataType>::countVerticeNumberForClipper(voxelVertNum, levelSet, TPlane3D<Real>(center, q.rotate(Coord(0, 1, 0))));

		Reduction<int> reduce;
		int totalVNum = reduce.accumulate(voxelVertNum.begin(), voxelVertNum.size());

		Scan<int> scan;
		scan.exclusive(voxelVertNum.begin(), voxelVertNum.size());

		this->stateField()->resize(totalVNum);

		DArray<Coord> vertices(totalVNum);

		DArray<TopologyModule::Triangle> triangles(totalVNum / 3);

		MarchingCubesHelper<TDataType>::constructTrianglesForClipper(
			this->stateField()->getData(),
			vertices, 
			triangles,
			voxelVertNum, 
			levelSet, 
			TPlane3D<Real>(center, q.rotate(Coord(0, 1, 0))));

		if (this->stateTriangleSet()->isEmpty()) {
			this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		}

		auto triSet = this->stateTriangleSet()->getDataPtr();
		triSet->setPoints(vertices);
		triSet->setTriangles(triangles);

		voxelVertNum.clear();
		vertices.clear();
		triangles.clear();
	}

	DEFINE_CLASS(VolumeClipper);
}