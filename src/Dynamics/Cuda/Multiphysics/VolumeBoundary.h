#pragma once
#include "Volume/Volume.h"

#include "ParticleSystem/ParticleSystem.h"
#include "Peridynamics/TriangularSystem.h"
#include "Peridynamics/TetrahedralSystem.h"

namespace dyno
{
	template <typename TDataType> class BoundaryConstraint;

	template<typename TDataType>
	class VolumeBoundary : public Node
	{
		DECLARE_TCLASS(VolumeBoundary, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VolumeBoundary();
		~VolumeBoundary() override;

		std::string getNodeType() override { return "Multiphysics"; }

	public:
		DEF_VAR(Real, NormalFriction, Real(0.95), "Normal friction");
		DEF_VAR(Real, TangentialFriction, Real(0), "Tangential friction");
		
	public:
		DEF_NODE_PORTS(Volume<TDataType>, Volume, "Level sets used as boundary");

		DEF_NODE_PORTS(ParticleSystem<TDataType>, ParticleSystem, "Particle Systems");

		DEF_NODE_PORTS(TriangularSystem<TDataType>, TriangularSystem, "Triangular Systems");

		DEF_NODE_PORTS(TetrahedralSystem<TDataType>, TetrahedralSystem, "Tetrahedral Systems");

		DEF_INSTANCE_STATE(TopologyModule, Topology, "");

	protected:
		void updateStates() override;

	private:
		void updateVolume();

		std::shared_ptr <BoundaryConstraint<TDataType>> mBoundaryConstraint;
	};

	IMPLEMENT_TCLASS(VolumeBoundary, TDataType)
}
