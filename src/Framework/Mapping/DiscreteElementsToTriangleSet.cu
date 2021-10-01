#include "DiscreteElementsToTriangleSet.h"

namespace dyno
{
	typedef typename TOrientedBox3D<Real> Box3D;

	template<typename TDataType>
	DiscreteElementsToTriangleSet<TDataType>::DiscreteElementsToTriangleSet()
		: TopologyMapping()
	{

	}

	template<typename Triangle>
	__global__ void SetupTriangles(
		DArray<Vec3f> vertices,
		DArray<Triangle> indices,
		DArray<Box3D> boxes)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boxes.size()) return;
		
		Box3D box = boxes[tId];

		Vec3f hx = box.u*box.extent[0];
		Vec3f hy = box.v*box.extent[1];
		Vec3f hz = box.w*box.extent[2];

		Vec3f hyz = hy + hz;
		Vec3f hxy = hx + hy;
		Vec3f hxz = hx + hz;

		Vec3f c = box.center;

		Vec3f v0 = c - hx - hyz;
		Vec3f v1 = c + hx - hyz;
		Vec3f v2 = c + hxz - hy;
		Vec3f v3 = c - hxy + hz;

		Vec3f v4 = c - hxz + hy;
		Vec3f v5 = c + hxy - hz;
		Vec3f v6 = c + hx + hyz;
		Vec3f v7 = c - hx + hyz;

		vertices[tId * 8] = v0;
		vertices[tId * 8 + 1] = v1;
		vertices[tId * 8 + 2] = v2;
		vertices[tId * 8 + 3] = v3;
		vertices[tId * 8 + 4] = v4;
		vertices[tId * 8 + 5] = v5;
		vertices[tId * 8 + 6] = v6;
		vertices[tId * 8 + 7] = v7;

		uint offset = tId * 8;

		indices[tId * 12] = Triangle(offset + 0, offset + 1, offset + 2);
		indices[tId * 12 + 1] = Triangle(offset + 0, offset + 2, offset + 3);

		indices[tId * 12 + 2] = Triangle(offset + 0, offset + 4, offset + 5);
		indices[tId * 12 + 3] = Triangle(offset + 0, offset + 5, offset + 1);

		indices[tId * 12 + 4] = Triangle(offset + 4, offset + 7, offset + 6);
		indices[tId * 12 + 5] = Triangle(offset + 4, offset + 6, offset + 5);

		indices[tId * 12 + 6] = Triangle(offset + 1, offset + 5, offset + 6);
		indices[tId * 12 + 7] = Triangle(offset + 1, offset + 6, offset + 2);

		indices[tId * 12 + 8] = Triangle(offset + 2, offset + 6, offset + 7);
		indices[tId * 12 + 9] = Triangle(offset + 2, offset + 7, offset + 3);

		indices[tId * 12 + 10] = Triangle(offset + 0, offset + 3, offset + 7);
		indices[tId * 12 + 11] = Triangle(offset + 0, offset + 7, offset + 4);
	}

	template<typename TDataType>
	bool DiscreteElementsToTriangleSet<TDataType>::apply()
	{
		if (this->outTriangleSet()->isEmpty())
		{
			this->outTriangleSet()->allocate();
		}

		auto inTopo = this->inDiscreteElements()->getDataPtr();

		//printf("====================================================== inside box update\n");
		auto& spheres = inTopo->getSpheres();
		auto& boxes = inTopo->getBoxes();
		auto& tets = inTopo->getTets();

		int numOfSpheres = 0;// m_spheres.size();
		int numOfBoxes = boxes.size();
		int numOfTets = tets.size();

		auto triSet = this->outTriangleSet()->getDataPtr();

		auto& points = triSet->getPoints();
		auto indices = triSet->getTriangles();

		int numOfVertices = 8 * numOfBoxes;
		int numOfTriangles = 12 * numOfBoxes;

		points.resize(numOfVertices);
		indices->resize(numOfTriangles);

		cuExecute(numOfBoxes,
			SetupTriangles,
			points,
			*indices,
			boxes);

		return true;
	}

	DEFINE_CLASS(DiscreteElementsToTriangleSet);
}