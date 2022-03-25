#include "HeightFieldToTriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	HeightFieldToTriangleSet<TDataType>::HeightFieldToTriangleSet()
		: TopologyMapping()
	{
	}

	template<typename Real, typename Coord>
	__global__ void SetupVerticesForHeightField(
		DArray<Coord> vertices,
		DArray2D<Coord> displacement,
		Coord origin,
		Real h,
		Coord translation,
		Real scale)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);

		if (i >= displacement.nx() || j >= displacement.ny()) return;

		Coord di = displacement(i, j);
		Coord v = Coord(origin.x + i * h + di.x, origin.y + di.y, origin.z + j * h + di.z);

		vertices[i + j * displacement.nx()] = scale * v + translation;
	}

	template<typename Triangle>
	__global__ void SetupTrianglesForHeightField(
		DArray<Triangle> vertices,
		uint nx,
		uint ny)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);

		if (i >= nx - 1 || j >= ny - 1) return;

		uint v0 = i + j * nx;
		uint v1 = i + 1 + j * nx;

		uint v2 = i + (j + 1) * nx;
		uint v3 = i + 1 + (j + 1) * nx;

		Triangle t0(v1, v0, v2);
		Triangle t1(v1, v2, v3);

		uint offset = 2 * i + 2 * j * (nx - 1);
		vertices[offset] = t0;
		vertices[offset + 1] = t1;
	}

	template<typename TDataType>
	bool HeightFieldToTriangleSet<TDataType>::apply()
	{
		if (this->outTriangleSet()->isEmpty())
		{
			this->outTriangleSet()->allocate();
		}

		auto heights = this->inHeightField()->getDataPtr();

		auto triSet = this->outTriangleSet()->getDataPtr();

		auto& vertices = triSet->getPoints();
		auto& indices = triSet->getTriangles();

		int numOfVertices = heights->width() * heights->height();
		int numOfTriangles = 2 * (heights->width() - 1) * (heights->height() - 1);

		vertices.resize(numOfVertices);
		indices.resize(numOfTriangles);


		auto& disp = heights->getDisplacement();

		Real scale = this->varScale()->getData();
		Coord translation = this->varTranslation()->getData();

		uint2 dim;
		dim.x = disp.nx();
		dim.y = disp.ny();
		cuExecute2D(dim,
			SetupVerticesForHeightField,
			vertices,
			disp,
			heights->getOrigin(),
			heights->getGridSpacing(),
			translation,
			scale);

		dim.x = disp.nx() - 1;
		dim.y = disp.ny() - 1;
		cuExecute2D(dim,
			SetupTrianglesForHeightField,
			indices,
			heights->width(),
			heights->height());

		return true;
	}

	DEFINE_CLASS(HeightFieldToTriangleSet);
}