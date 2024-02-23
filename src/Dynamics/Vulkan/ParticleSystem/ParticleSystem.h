#pragma once
#include "Node.h"

#include "Topology/PointSet.h"

namespace dyno
{
	/*!
	*	\class	ParticleSystem
	*	\brief	This class represents the base class for more advanced particle-based nodes.
	*/
	class ParticleSystem : public Node
	{
		DECLARE_CLASS(ParticleSystem)
	public:
		ParticleSystem();
		~ParticleSystem() override;

		std::string getNodeType() override;

		float getDt() override { return 0.001f; }
		
	public:
		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Vec3f, Position, DeviceType::GPU, "Particle position");

		/**
		 * @brief Particle velocity
		 */
		DEF_ARRAY_STATE(Vec3f, Velocity, DeviceType::GPU, "Particle velocity");

		/**
		 * @brief Particle force
		 */
		DEF_ARRAY_STATE(Vec3f, Force, DeviceType::GPU, "Force on each particle");

		/**
		 * @brief A topology
		 */
		DEF_INSTANCE_STATE(PointSet3f, PointSet, "Topology");

	protected:
		void resetStates() override;
	};
}