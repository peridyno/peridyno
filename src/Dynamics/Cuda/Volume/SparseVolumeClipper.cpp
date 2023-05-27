#include "SparseVolumeClipper.h"

#include "Mapping/MarchingCubesHelper.h"

#include "VolumeHelper.h"

namespace dyno
{
	template<typename TDataType>
	SparseVolumeClipper<TDataType>::SparseVolumeClipper()
		: Node()
	{
	}

	template<typename TDataType>
	SparseVolumeClipper<TDataType>::~SparseVolumeClipper()
	{
	}

	template<typename TDataType>
	void SparseVolumeClipper<TDataType>::resetStates()
	{
		auto sv = this->getSparseVolume();

		if (sv->stateSDFTopology()->isEmpty())
		{
			printf("SparseMarchingCubes: The import is empty! \n");
			return;
		}

		auto center = this->varTranslation()->getData();
		auto eulerAngles = this->varRotation()->getData();

		Quat<Real> q = Quat<Real>::fromEulerAngles(eulerAngles[0], eulerAngles[1], eulerAngles[2]);

		auto octree = sv->stateSDFTopology()->getDataPtr();

		DArray<Coord> ceilVertices;

		octree->getCellVertices(ceilVertices);

		DArray<Real> sdfs;
		DArray<Coord> normals;

		octree->getSignDistance(ceilVertices, sdfs, normals);
		//sv->getSignDistanceMLS(ceilVertices, sdfs, normals);
		//sv->getSignDistanceKernel(ceilVertices, sdfs);

		//DArray3D<Real> distances(nx + 1, ny + 1, nz + 1);
		DArray<uint> voxelVertNum(ceilVertices.size() / 8);

		MarchingCubesHelper<TDataType>::countVerticeNumberForOctreeClipper(
			voxelVertNum,
			ceilVertices,
			TPlane3D<Real>(center, q.rotate(Coord(0, 1, 0))));

		Reduction<uint> reduce;
		uint totalVNum = reduce.accumulate(voxelVertNum.begin(), voxelVertNum.size());

		Scan<uint> scan;
		scan.exclusive(voxelVertNum.begin(), voxelVertNum.size());

		DArray<Coord> triangleVertices(totalVNum);

		DArray<TopologyModule::Triangle> triangles(totalVNum / 3);

		this->stateField()->resize(totalVNum);

		MarchingCubesHelper<TDataType>::constructTrianglesForOctreeClipper(
			this->stateField()->getData(),
			triangleVertices,
			triangles,
			voxelVertNum,
			ceilVertices,
			sdfs,
			TPlane3D<Real>(center, q.rotate(Coord(0, 1, 0))));

		if (this->stateTriangleSet()->isEmpty()) {
			this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		}

		auto triSet = this->stateTriangleSet()->getDataPtr();
		triSet->setPoints(triangleVertices);
		triSet->setTriangles(triangles);

		this->stateVertices()->assign(triangleVertices);

		sdfs.clear();
		normals.clear();
		voxelVertNum.clear();
		ceilVertices.clear();
		triangleVertices.clear();
		triangles.clear();
	}

	template<typename TDataType>
	void SparseVolumeClipper<TDataType>::updateStates()
	{
		this->reset();
	}

	DEFINE_CLASS(SparseVolumeClipper);
}