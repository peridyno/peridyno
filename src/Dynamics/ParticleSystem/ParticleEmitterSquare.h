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
	class ParticleEmitterSquare : public ParticleEmitter<TDataType>
	{
		DECLARE_CLASS_1(ParticleEmitterSquare, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleEmitterSquare(std::string name = "particleEmitter");
		virtual ~ParticleEmitterSquare();

		void generateParticles() override;

		//void advance(Real dt) override;
	private:
		DEF_VAR(Width, Real, 0.05, "Emitter width");
		DEF_VAR(Height, Real, 0.05, "Emitter height");
	};
}