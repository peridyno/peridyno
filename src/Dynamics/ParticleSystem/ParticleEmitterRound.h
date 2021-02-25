#pragma once
#include "ParticleSystem.h"
#include "ParticleEmitter.h"

namespace dyno
{
	/*!
	*	\class	ParticleFluid
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class ParticleEmitterRound : public ParticleEmitter<TDataType>
	{
		DECLARE_CLASS_1(ParticleEmitterRound, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleEmitterRound(std::string name = "particleEmitter");
		virtual ~ParticleEmitterRound();

		void generateParticles() override;
		
		
		//void advance(Real dt) override;
	public:
		DEF_VAR(Radius, Real, 0.05, "Emitter radius");

		//DEF_NODE_PORTS(ParticleSystems, ParticleSystem<TDataType>, "Particle Systems");
	};

#ifdef PRECISION_FLOAT
	template class ParticleEmitterRound<DataType3f>;
#else
	template class ParticleEmitterRound<DataType3d>;
#endif
}