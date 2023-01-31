#pragma once
#include "RigidBody/RigidBody.h"
#include "Topology/TriangleSet.h"

#include "FilePath.h"

namespace dyno 
{
	template<typename TDataType>
	class RigidMesh : public RigidBody<TDataType>
	{
		DECLARE_TCLASS(RigidMesh, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		RigidMesh();
		~RigidMesh();

	public:
		DEF_VAR(FilePath, EnvelopeName, "", "");

		DEF_VAR(FilePath, MeshName, "", "");

		DEF_VAR(Coord, Location, 0, "Location");
		DEF_VAR(Coord, Rotation, 0, "Rotation");
		DEF_VAR(Coord, Scale, Coord(1), "Scale");

		DEF_VAR(Real, Density, Real(1000), "Density");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, InitialEnvelope, "Envelope for the mesh");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, Envelope, "Envelope for the mesh");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, InitialMesh, "Initial mesh");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, Mesh, "Mesh");

	protected:
		void resetStates() override;

		void updateStates() override;
	};

	IMPLEMENT_TCLASS(RigidMesh, TDataType)
}
