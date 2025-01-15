#include "ApplyBumpMap2TriangleSet.h"

#include "Math/Lerp.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ApplyBumpMap2TriangleSet, TDataType)

	template<typename TDataType>
	ApplyBumpMap2TriangleSet<TDataType>::ApplyBumpMap2TriangleSet()
		: TopologyMapping()
	{
	}

	template<typename Real, typename Coord3D>
	__global__ void ABMTS_UpdateVertices(
		DArray<Coord3D> vertices,
		DArray2D<Coord3D> bumpMap,
		Coord3D origin,
		Real h)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= vertices.size()) return;

		Coord3D xyz = vertices[tId];

		Real x = (xyz.x - origin.x) / h;
		Real z = (xyz.z - origin.z) / h;

		Coord3D disp = bilinear(bumpMap, x, z);

		vertices[tId] = xyz + disp;
	}

	template<typename TDataType>
	bool ApplyBumpMap2TriangleSet<TDataType>::apply()
	{
		if (this->outTriangleSet()->isEmpty()) {
			this->outTriangleSet()->allocate();
		}

		auto inTs = this->inTriangleSet()->constDataPtr();
		auto outTs = this->outTriangleSet()->getDataPtr();

		auto topo = this->inHeightField()->constDataPtr();
		auto& disp = topo->getDisplacement();
		Coord3D origin = topo->getOrigin();
		Real h = topo->getGridSpacing();
		
		outTs->copyFrom(*inTs);

		auto& vertices = outTs->getPoints();

		cuExecute(vertices.size(),
			ABMTS_UpdateVertices,
			vertices,
			disp,
			origin,
			h);

		outTs->update();

		return true;
	}

	DEFINE_CLASS(ApplyBumpMap2TriangleSet);
}