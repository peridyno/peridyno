#pragma once
#include "Module/ComputeModule.h"

#include "Algorithm/Reduction.h"

#include "Topology/SparseOctree.h"

namespace dyno
{
	typedef typename ::dyno::TAlignedBox3D<Real> AABB;

	template<typename TDataType>
	class SparseNeighborPointQuery : public ComputeModule
	{
		DECLARE_TCLASS(SparseNeighborPointQuery, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SparseNeighborPointQuery();
		virtual ~SparseNeighborPointQuery();

		void compute() override;

	public:
		DEF_VAR_IN(Real, Radius, "Search radius");

		DEF_ARRAY_IN(Coord, Source, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Target, DeviceType::GPU, "");

		DEF_ARRAYLIST_OUT(int, NeighborIds, DeviceType::GPU, "Contact pairs");

	private:
		Reduction<Real> m_reduce_real;
		Reduction<Coord> m_reduce_coord;

		SparseOctree<TDataType> octree;
	};
}
