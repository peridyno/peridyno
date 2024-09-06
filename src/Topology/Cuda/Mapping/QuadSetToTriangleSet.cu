#include "QuadSetToTriangleSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(QuadSetToTriangleSet, TDataType)

	template<typename TDataType>
	QuadSetToTriangleSet<TDataType>::QuadSetToTriangleSet()
		: TopologyMapping()
	{
	}

	template<typename Triangle, typename Quad>
	__global__ void Q2T_SetupTriangles(
		DArray<Triangle> triangles,
		DArray<Quad> quads)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= quads.size()) return;

		Quad quad = quads[tId];
		triangles[2 * tId] = Triangle(quad[0], quad[1], quad[2]);
		triangles[2 * tId + 1] = Triangle(quad[0], quad[2], quad[3]);
	}

	template<typename TDataType>
	bool QuadSetToTriangleSet<TDataType>::apply()
	{
		if (this->outTriangleSet()->isEmpty()) {
			this->outTriangleSet()->allocate();
		}

		auto qs = this->inQuadSet()->constDataPtr();
		auto ts = this->outTriangleSet()->getDataPtr();

		auto& verts = qs->getPoints();
		auto& quads = qs->getQuads();

		auto& tris = ts->getTriangles();
		tris.resize(2 * quads.size());

		ts->setPoints(verts);

		cuExecute(quads.size(),
			Q2T_SetupTriangles,
			tris,
			quads);

		this->outTriangleSet()->update();

		return true;
	}

	DEFINE_CLASS(QuadSetToTriangleSet);
}