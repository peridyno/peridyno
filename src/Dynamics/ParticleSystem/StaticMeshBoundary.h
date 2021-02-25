#pragma once
#include "Framework/Node.h"
#include "Framework/FieldArray.h"

#include "RigidBody/RigidBody.h"
#include "ParticleSystem/ParticleSystem.h"

namespace dyno {

	template <typename T> class TriangleSet;
	template <typename T> class NeighborTriangleQuery;

	template<typename TDataType>
	class StaticMeshBoundary : public Node
	{
		DECLARE_CLASS_1(StaticMeshBoundary, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		StaticMeshBoundary();
		~StaticMeshBoundary() override;


		void loadMesh(std::string filename);

		void advance(Real dt) override;

		bool initialize() override;

		bool resetStatus() override;

	public:
		DEF_NODE_PORTS(RigidBody, RigidBody<TDataType>, "A rigid body");
		DEF_NODE_PORTS(ParticleSystem, ParticleSystem<TDataType>, "Particle Systems");


	public:
		/**
		 * @brief Particle position
		 */
		DEF_EMPTY_CURRENT_ARRAY(ParticlePosition, Coord, DeviceType::GPU, "Particle position");


		/**
		 * @brief Particle velocity
		 */
		DEF_EMPTY_CURRENT_ARRAY(ParticleVelocity, Coord, DeviceType::GPU, "Particle velocity");

		/**
		 * @brief Triangle vertex
		 */
		DEF_EMPTY_CURRENT_ARRAY(TriangleVertex, Coord, DeviceType::GPU, "Particle position");

		/**
		 * @brief Particle velocity
		 */
		DEF_EMPTY_CURRENT_ARRAY(TriangleIndex, Triangle, DeviceType::GPU, "Particle velocity");

	private:
		std::shared_ptr<NeighborTriangleQuery<TDataType>> m_nbrQuery;
		VarField<Real> radius;

		std::vector<std::shared_ptr<TriangleSet<TDataType>>> m_obstacles;
	};


#ifdef PRECISION_FLOAT
template class StaticMeshBoundary<DataType3f>;
#else
template class StaticMeshBoundary<DataType3d>;
#endif

}
