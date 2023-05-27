#pragma once
#include "Node.h"

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

		void translate(Coord t);

		std::shared_ptr<Node> loadSDF(std::string filename, bool bOutBoundary = false);
		std::shared_ptr<Node> loadCube(Coord lo, Coord hi, Real distance = 0.005f, bool bOutBoundary = false);

		void loadShpere(Coord center, Real r, Real distance = 0.005f, bool bOutBoundary = false, bool bVisible = false);

	public:
		DEF_VAR(Real, TangentialFriction, 0, "Tangential friction");
		DEF_VAR(Real, NormalFriction, 0, "Normal friction");

		DEF_NODE_PORTS(ParticleSystem<TDataType>, ParticleSystem, "Particle Systems");

		DEF_NODE_PORTS(TriangularSystem<TDataType>, TriangularSystem, "Triangular Systems");

		DEF_NODE_PORTS(TetrahedralSystem<TDataType>, TetrahedralSystem, "Tetrahedral Systems");

		DEF_INSTANCE_STATE(TopologyModule, Topology, "");

	protected:
		void updateVolume();

		void updateStates() override;

	private:
		std::vector<std::shared_ptr<BoundaryConstraint<TDataType>>> m_obstacles;
	};

	IMPLEMENT_TCLASS(VolumeBoundary, TDataType)
}
