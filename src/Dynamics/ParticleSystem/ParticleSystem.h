#pragma once
#include "Node.h"
//#include "PointRenderModule.h"

namespace dyno
{
	template <typename TDataType> class PointSet;
	/*!
	*	\class	ParticleSystem
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class ParticleSystem : public Node
	{
		DECLARE_CLASS_1(ParticleSystem, TDataType)
	public:

		bool self_update = true;
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleSystem(std::string name = "default");
		virtual ~ParticleSystem();

		void loadParticles(Coord lo, Coord hi, Real distance);
		void loadParticles(Coord center, Real r, Real distance);
		void loadParticles(std::string filename);

		virtual bool translate(Coord t);
		virtual bool scale(Real s);

	public:
//		std::shared_ptr<PointRenderModule> getRenderModule();

		/**
		 * @brief Particle position
		 */
		DEF_EMPTY_CURRENT_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");


		/**
		 * @brief Particle velocity
		 */
		DEF_EMPTY_CURRENT_ARRAY(Velocity, Coord, DeviceType::GPU, "Particle velocity");

		/**
		 * @brief Particle force
		 */
		DEF_EMPTY_CURRENT_ARRAY(Force, Coord, DeviceType::GPU, "Force on each particle");

		
	protected:
		void updateTopology() override;
		void resetStates() override;
//		virtual void setVisible(bool visible) override;
	};
}