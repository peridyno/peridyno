#pragma once
#include "Node.h"

#include "StaticTriangularMesh.h"

#include "ParticleSystem/Attribute.h"

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
		
		SemiAnalyticalSFINode(std::string name = "SFINode");
		~SemiAnalyticalSFINode() override;

	public:
		bool surfaceTensionSet(Real i) { this->varSurfaceTension()->setValue(i); return true; };
		bool AdhesionIntensitySet(Real i) { this->varAdhesionIntensity()->setValue(i); return true; };
		bool RestDensitySet(Real i) { this->varRestDensity()->setValue(i); return true; };

		DEF_NODE_PORTS(ParticleSystem<TDataType>, ParticleSystem, "Particle Systems");
		DEF_NODE_PORTS(StaticTriangularMesh<TDataType>, BoundaryMesh, "Triangular Surface Mesh Boundary");
		
	public:

		DEF_VAR(Bool, Fast, false, "");
		DEF_VAR(Real, Radius, Real(0.0125), "radius");

		DEF_VAR(Real, SurfaceTension, Real(0.055), "surface tension");
		DEF_VAR(Real, AdhesionIntensity, Real(30.0), "adhesion");
		DEF_VAR(Real, RestDensity, Real(1000), "Rest Density");

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
	};

}