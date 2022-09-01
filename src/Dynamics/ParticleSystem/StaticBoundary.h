#pragma once
#include "Node.h"
#include "RigidBody/RigidBody.h"

#include "ParticleSystem.h"
#include "FilePath.h"


namespace dyno {
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

		void loadSDF(std::string filename, bool bOutBoundary = false);
		std::shared_ptr<Node> loadCube(Coord lo, Coord hi, Real distance = 0.005f, bool bOutBoundary = false);
		void loadShpere(Coord center, Real r, Real distance = 0.005f, bool bOutBoundary = false, bool bVisible = false);

		void translate(Coord t);
		void scale(Real s);

		void resetStates();
	protected:
		void updateStates() override;

	public:
		DEF_VAR(Real, TangentialFriction, 0.0, "Tangential friction");
		DEF_VAR(Real, NormalFriction, 1.0, "Normal friction");

		DEF_VAR(Vec3f, CubeVertex_lo, Vec3f(0.0f), "Cube");
		DEF_VAR(Vec3f, CubeVertex_hi, Vec3f(1.0f), "Cube");

		DEF_VAR(FilePath, FileName, "", "");

		std::vector<std::shared_ptr<BoundaryConstraint<TDataType>>> m_obstacles;

		std::vector<std::shared_ptr<RigidBody<TDataType>>> m_rigids;
		std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems;

		DEF_NODE_PORTS(RigidBody<TDataType>, RigidBody, "A rigid body");
		DEF_NODE_PORTS(ParticleSystem<TDataType>, ParticleSystem, "Particle Systems");
	};
}
