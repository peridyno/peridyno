#include "VolumeClipper.h"

#include "Module/MarchingCubesHelper.h"

#include "ColorMapping.h"
#include "GLSurfaceVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	VolumeClipper<TDataType>::VolumeClipper()
		: Node()
	{
		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMin()->setValue(-0.5);
		colorMapper->varMax()->setValue(0.5);
		this->stateField()->connect(colorMapper->inScalar());
		this->graphicsPipeline()->pushModule(colorMapper);
		// 
		// 
		auto surfaceVisualizer = std::make_shared<GLSurfaceVisualModule>();
		surfaceVisualizer->varColorMode()->getDataPtr()->setCurrentKey(1);
		colorMapper->outColor()->connect(surfaceVisualizer->inColor());
		this->stateTriangleSet()->connect(surfaceVisualizer->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfaceVisualizer);
	}

	template<typename TDataType>
	VolumeClipper<TDataType>::~VolumeClipper()
	{
	}

	template<typename TDataType>
	void VolumeClipper<TDataType>::resetStates()
	{
		auto center = this->varTranslation()->getValue();
		auto eulerAngles = this->varRotation()->getValue();

		eulerAngles /= 180.0f;
		eulerAngles *= M_PI;

		auto levelSet = this->inLevelSet()->getDataPtr()->getSDF();

		Coord lo = levelSet.lowerBound();
		Coord hi = levelSet.upperBound();

		Quat<Real> q = Quat<Real>::fromEulerAngles(eulerAngles[0], eulerAngles[1], eulerAngles[2]);

		int nx = levelSet.nx() - 1;
		int ny = levelSet.ny() - 1;
		int nz = levelSet.nz() - 1;

		DArray<int> voxelVertNum(nx * ny * nz);

		Coord shifted_center = center + 0.5 * (lo + hi);

		MarchingCubesHelper<TDataType>::countVerticeNumberForClipper(voxelVertNum, levelSet, TPlane3D<Real>(shifted_center, q.rotate(Coord(0, 1, 0))));

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
			TPlane3D<Real>(shifted_center, q.rotate(Coord(0, 1, 0))));

		auto triSet = this->stateTriangleSet()->getDataPtr();
		triSet->setPoints(vertices);
		triSet->setTriangles(triangles);
		triSet->update();

		voxelVertNum.clear();
		vertices.clear();
		triangles.clear();
	}

	DEFINE_CLASS(VolumeClipper);
}