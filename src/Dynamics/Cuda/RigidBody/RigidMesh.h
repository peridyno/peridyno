#pragma once
#include "RigidBody/RigidBody.h"
#include "Topology/TriangleSet.h"

#include "FilePath.h"

namespace dyno 
{
	template<typename TDataType>
	class RigidMesh : virtual public RigidBody<TDataType>
	{
		DECLARE_TCLASS(RigidMesh, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		RigidMesh();
		~RigidMesh() override;

	public:
		DEF_VAR(FilePath, EnvelopeName, getAssetPath() + "obj/boat_boundary.obj", "");

		DEF_VAR(FilePath, MeshName, getAssetPath() + "obj/boat_mesh.obj", "");

		DEF_VAR(Real, Density, Real(1000), "Density");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, InitialEnvelope, "Envelope for the mesh");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, Envelope, "Envelope for the mesh");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, InitialMesh, "Initial mesh");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, Mesh, "Mesh");

	protected:
		void resetStates() override;

		void updateStates() override;

	private:
		void transform();
	};

	IMPLEMENT_TCLASS(RigidMesh, TDataType)
}
