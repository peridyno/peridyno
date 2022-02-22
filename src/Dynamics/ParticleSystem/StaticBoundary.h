#pragma once
#include "Node.h"
#include "RigidBody/RigidBody.h"
#include "ParticleSystem/ParticleSystem.h"

namespace dyno {
// 	template <typename TDataType> class RigidBody;
// 	template <typename TDataType> class ParticleSystem;
	template <typename TDataType> class DistanceField3D;
	template <typename TDataType> class BoundaryConstraint;

	template<typename TDataType>
	class StaticBoundary : public Node
	{
		DECLARE_TCLASS(StaticBoundary, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		StaticBoundary();
		~StaticBoundary() override;

//		bool addRigidBody(std::shared_ptr<RigidBody<TDataType>> child);
//		bool addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child);

		void loadSDF(std::string filename, bool bOutBoundary = false);
		std::shared_ptr<Node> loadCube(Coord lo, Coord hi, Real distance = 0.005f, bool bOutBoundary = false);
		void loadShpere(Coord center, Real r, Real distance = 0.005f, bool bOutBoundary = false, bool bVisible = false);

		void translate(Coord t);
		void scale(Real s);

	protected:
		void updateStates() override;

	public:
		DEF_VAR(Real, TangentialFriction, 0.0, "Tangential friction");
		DEF_VAR(Real, NormalFriction, 1.0, "Normal friction");


		std::vector<std::shared_ptr<BoundaryConstraint<TDataType>>> m_obstacles;

		std::vector<std::shared_ptr<RigidBody<TDataType>>> m_rigids;
		std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;

		DEF_NODE_PORTS(RigidBody, RigidBody<TDataType>, "A rigid body");
		DEF_NODE_PORTS(ParticleSystem, ParticleSystem<TDataType>, "Particle Systems");
	};
}
