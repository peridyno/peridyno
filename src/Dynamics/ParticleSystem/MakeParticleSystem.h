#pragma once
#include "ParticleSystem.h"

namespace dyno
{
	/*!
	*	\class	ParticleSystem
	*	\brief	This class represents the base class for more advanced particle-based nodes.
	*/
	template<typename TDataType>
	class MakeParticleSystem : public ParticleSystem<TDataType>
	{
		DECLARE_TCLASS(ParticleSystem, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		MakeParticleSystem();
		virtual ~MakeParticleSystem();

		DEF_VAR(Coord, InitialVelocity, Coord(0.0f), "Initial Particle Velocity");

		DEF_INSTANCE_IN(PointSet<TDataType>, Points, "");
	protected:
		void resetStates() override;
	};

	IMPLEMENT_TCLASS(MakeParticleSystem, TDataType)
}