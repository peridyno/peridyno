#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Collision/NeighborPointQuery.h"
#include "Collision/NeighborTriangleQuery.h"
#include "Topology/TriangleSet.h"
#include "SemiAnalyticalScheme/TriangularMeshConstraint.h"
#include "Auxiliary/DataSource.h"

namespace dyno
{
	template <typename TDataType>
	class ParticleMeshCollidingNode : public Node
	{
		DECLARE_TCLASS(SemiAnalyticalSFINode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;


		//DEF_NODE_PORTS(ParticleSystem<TDataType>, ParticleSystem, "Particle Systems");
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "Boundary triangular surface");

		ParticleMeshCollidingNode();
		~ParticleMeshCollidingNode() ;

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Particle position");
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Particle velocity");
		DEF_ARRAY_STATE(Triangle, TriangleIndex, DeviceType::GPU, "triangle_index");
		DEF_ARRAY_STATE(Coord, TriangleVertex, DeviceType::GPU, "triangle_vertex");

	protected:
		void resetStates() override;


		//void preUpdateStates() override;
		//void postUpdateStates() override;

	};

	IMPLEMENT_TCLASS(ParticleMeshCollidingNode, TDataType)


}
