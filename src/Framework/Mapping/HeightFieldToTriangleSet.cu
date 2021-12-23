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
		DArray2D<Real> height,
		Coord origin,
		Real dx,
		Real dz)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);

		if (i >= height.nx() || j >= height.ny()) return;

		Real h = height(i, j);

		Coord v = Coord(origin.x + i * dx, origin.y + h, origin.z + j * dz);

		vertices[i + j * height.nx()] = v;
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

		int numOfVertices = heights->length() * heights->width();
		int numOfTriangles = 2 * (heights->length() - 1) * (heights->width() - 1);

		vertices.resize(numOfVertices);
		indices.resize(numOfTriangles);


		auto height = heights->getHeights();

		uint2 dim;
		dim.x = height.nx();
		dim.y = height.ny();
		cuExecute2D(dim,
			SetupVerticesForHeightField,
			vertices,
			height,
			heights->getOrigin(),
			heights->getDx(),
			heights->getDz());

		dim.x = height.nx() - 1;
		dim.y = height.ny() - 1;
		cuExecute2D(dim,
			SetupTrianglesForHeightField,
			indices,
			heights->length(),
			heights->width());

		return true;
	}

	DEFINE_CLASS(HeightFieldToTriangleSet);
}