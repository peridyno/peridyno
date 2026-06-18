#include "SparseVolumeClipper.h"

#include "Module/MarchingCubesHelper.h"

#include "ColorMapping.h"
#include "GLSurfaceVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	SparseVolumeClipper<TDataType>::SparseVolumeClipper()
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
	SparseVolumeClipper<TDataType>::~SparseVolumeClipper()
	{
	}

	template<typename TDataType>
	void SparseVolumeClipper<TDataType>::resetStates()
	{
		auto& m_sdf = this->inAGridSDF()->getData();

		auto m_AGrid = this->inAGridSet()->getDataPtr();
		auto& nodes = m_AGrid->adaptiveGridNode();
		DArray<Coord> vertex;
		DArray<int> vertex_neighbor, node2ver;
		m_AGrid->extractVertex(vertex, vertex_neighbor, node2ver);
		vertex.clear();
		//auto& vertex_neighbor = m_AGrid->getVertexNeighbor();
		//auto& node2ver = m_AGrid->getNode2Vertex();
		//Real m_dx = m_AGrid->getDx();
		//Level m_levelmax = m_AGrid->getLevelMax();

		auto center = this->varTranslation()->getData();
		auto eulerAngles = this->varRotation()->getData();

		Quat<Real> q = Quat<Real>::fromEulerAngles(eulerAngles[0], eulerAngles[1], eulerAngles[2]);

		DArray<uint> voxelVertNum(nodes.size());
		voxelVertNum.reset();
		MarchingCubesHelper<TDataType>::countVerticeNumberForOctreeClipper(
			voxelVertNum,
			nodes,
			m_AGrid,
			TPlane3D<Real>(center, q.rotate(Coord(0, 1, 0))));

		Reduction<uint> reduce;
		uint totalVNum = reduce.accumulate(voxelVertNum.begin(), voxelVertNum.size());

		Scan<uint> scan;
		scan.exclusive(voxelVertNum.begin(), voxelVertNum.size());

		DArray<Coord> triangleVertices(totalVNum);

		DArray<Topology::Triangle> triangles(totalVNum / 3);

		this->stateField()->resize(totalVNum);

		MarchingCubesHelper<TDataType>::constructTrianglesForOctreeClipper(
			this->stateField()->getData(),
			triangleVertices,
			triangles,
			voxelVertNum,
			nodes,
			m_AGrid,
			m_sdf,
			TPlane3D<Real>(center, q.rotate(Coord(0, 1, 0))));

		if (this->stateTriangleSet()->isEmpty()) {
			this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		}

		auto triSet = this->stateTriangleSet()->getDataPtr();
		triSet->setPoints(triangleVertices);
		triSet->setTriangles(triangles);

		voxelVertNum.clear();
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