#pragma once
#include "Node.h"

namespace dyno
{
	template <typename TDataType> class PointSet;
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

		ParticleSystem(std::string name = "default");
		virtual ~ParticleSystem();

		void loadParticles(Coord lo, Coord hi, Real distance);
		void loadParticles(Coord center, Real r, Real distance);
		void loadParticles(std::string filename);

		virtual bool translate(Coord t);
		virtual bool scale(Real s);

		std::string getNodeType() override;

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

	protected:
		void updateTopology() override;
		void resetStates() override;
//		virtual void setVisible(bool visible) override;
	};
}