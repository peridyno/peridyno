#pragma once
#include "Node.h"

#include "Topology/PointSet.h"

namespace dyno
{
	/*!
	*	\class	ParticleSystem
	*	\brief	This class represents the base class for more advanced particle-based nodes.
	*/
	template<typename TDataType>
	class ParticleSystem : public Node
	{
		DECLARE_TCLASS(ParticleSystem, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleSystem();
		~ParticleSystem() override;

 		void loadParticles(Coord lo, Coord hi, Real distance);

		std::string getNodeType() override;

		Real getDt() override { return 0.001; }
		
	public:
		/**
		 * @brief Particle position
		 */
		DEF_ARRAY_STATE(Coord, Position, DeviceType::GPU, "Particle position");


		/**
		 * @brief Particle velocity
		 */
		DEF_ARRAY_STATE(Coord, Velocity, DeviceType::GPU, "Particle velocity");

		/**
		 * @brief Particle force
		 */
		DEF_ARRAY_STATE(Coord, Force, DeviceType::GPU, "Force on each particle");

		/**
		 * @brief A topology
		 */
		DEF_INSTANCE_STATE(PointSet<TDataType>, PointSet, "Topology");

	protected:
		void resetStates() override;

		void postUpdateStates() override;
	};
}