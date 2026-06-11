#pragma once
#include "Module/ComputeModule.h"

#include "Topology/TriangleSet.h"

#include "Primitive/Primitive3D.h"

namespace dyno
{
	template<typename TDataType> class CollisionDetectionBroadPhase;

	template<typename TDataType>
	class DynamicNeighborTriangleQuery : public ComputeModule
	{
		DECLARE_TCLASS(DynamicNeighborTriangleQuery, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename Topology::Triangle Triangle;
		typedef typename TAlignedBox3D<Real> AABB;

		DynamicNeighborTriangleQuery();
		~DynamicNeighborTriangleQuery() override;

		void compute() override;

	public:
		DECLARE_ENUM(Spatial,
		BVH = 0,
			OCTREE = 1);

		DEF_ENUM(Spatial, Spatial, Spatial::BVH, "");

		/**
		* @brief Search radius
		* A positive value representing the radius of neighborhood for each point
		*/
		DEF_VAR_IN(Real, Radius, "Search radius");

		DEF_VAR_IN(Real, TimeStep, "Time step size");
		DEF_VAR(Vec3f, Gravity, Vec3f(0.0f, -9.8f, 0.0f), "Gravity");
		/**
		 * @brief A set of points to be required from.
		 */
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "A set of points to be required from");

		/**
		* @brief Velocity
		* Particle velocity
		*/
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Particle velocity");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "A set of triangles to be required from");

		/**
		 * @brief A set of points to be required from.
		 */
		DEF_ARRAY_IN(Coord, PreTriPosition, DeviceType::GPU, "A set of PreTriangles to be required from");

		/**
		 * @brief Ids of neighboring particles
		 */
		DEF_ARRAYLIST_OUT(int, NeighborIds, DeviceType::GPU, "Return neighbor ids");

	private:
		DArray<AABB> mQueryAABB;
		DArray<AABB> mQueriedAABB;

		Reduction<uint> mReduce;

		std::shared_ptr<CollisionDetectionBroadPhase<TDataType>> mBroadPhaseCD;
	};
}