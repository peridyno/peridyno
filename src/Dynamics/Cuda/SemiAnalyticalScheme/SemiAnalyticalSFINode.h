#pragma once
#include "Node.h"

#include "Topology/TriangleSet.h"

#include "Collision/Attribute.h"

namespace  dyno
{
	template <typename T> class ParticleSystem;
	/*!
	*	\class	SemiAnalyticalSFINode
	*	\brief	Semi-Analytical Solid Fluid Interaction
	*
	*	This class defines all fields necessary to implement a one way coupling between particles and static boundary meshes.
	*
	*/

	template<typename TDataType>
	class SemiAnalyticalSFINode : public Node
	{
		DECLARE_TCLASS(SemiAnalyticalSFINode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		
		SemiAnalyticalSFINode();
		~SemiAnalyticalSFINode() override;

	public:
		DEF_NODE_PORTS(ParticleSystem<TDataType>, ParticleSystem, "Particle Systems");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "Boundary triangular surface");
		
	public:
		DEF_VAR(Bool, Fast, false, "");
		DEF_VAR(Bool, SyncBoundary, false, "");

		DEF_ARRAY_STATE(Coord, Position, DeviceType::GPU, "Particle position");
		DEF_ARRAY_STATE(Coord, Velocity, DeviceType::GPU, "Particle velocity");
		DEF_ARRAY_STATE(Coord, ForceDensity, DeviceType::GPU, "Force density");

		DEF_ARRAY_STATE(Attribute, Attribute, DeviceType::GPU, "Particle attribute");

		DEF_ARRAY_STATE(Triangle, TriangleIndex, DeviceType::GPU, "triangle_index");
		DEF_ARRAY_STATE(Coord, TriangleVertex, DeviceType::GPU, "triangle_vertex");
		
	protected:
		void resetStates() override;

		void preUpdateStates() override;
		void postUpdateStates() override;

		bool validateInputs() override;
	};
}