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

		ParticleSystem(std::string name = "default");
		virtual ~ParticleSystem();

		void loadParticles(Coord lo, Coord hi, Real distance);
		void loadParticles(Coord center, Real r, Real distance);
		void loadParticles(Coord center, Real r, Coord center1, Real r1, Real distance);
		void loadParticles(Coord center, Real r,
			Coord center1, Real r1,
			Coord center2, Real r2,
			Coord center3, Real r3,
			Coord lo, Coord hi, Real distance);
		void loadParticles(Coord center, Real r, 
			Coord center1, Real r1,
			Coord center2, Real r2,
			Coord center3, Real r3,
			Coord center4, Real r4,
			Coord center5, Real r5,
			Coord center6, Real r6,
			Coord center7, Real r7,
			Coord center8, Real r8,
			Coord center9, Real r9,
			Coord center10, Real r10,
			Coord center11, Real r11,
			Coord center12, Real r12,
			Coord center13, Real r13,
			Coord center14, Real r14, Real distance);
		void loadParticles(std::string filename);

		virtual bool translate(Coord t);
		virtual bool scale(Real s);
		virtual bool rotate(Quat<Real> q);
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

		/**
		 * @brief A topology
		 */
		DEF_INSTANCE_STATE(PointSet<TDataType>, PointSet, "Topology");

	protected:
		void updateTopology() override;
		void resetStates() override;
//		virtual void setVisible(bool visible) override;
		DArray<Coord> positionLast;
	};
}