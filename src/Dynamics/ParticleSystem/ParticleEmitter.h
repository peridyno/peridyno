#pragma once
#include "ParticleSystem.h"

namespace dyno
{
	/*!
	*	\class	ParticleEimitter
	*	\brief	
	*/
	template <typename T> class ParticleFluid;
	template<typename TDataType>
	class ParticleEmitter : public ParticleSystem<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleEmitter(std::string name = "particle emitter");
		virtual ~ParticleEmitter();

		void advance2(Real dt);
		void advance(Real dt) override;
		virtual void generateParticles();

		void updateTopology() override;
		bool resetStatus() override;


		//DEF_VAR(Vec3f, Centre, 0, "Emitter location");
		//DEF_VAR(Real, Radius, 0.1, "Emitter scale");
		DEF_VAR(Real, VelocityMagnitude, 1, "Emitter Velocity");
		DEF_VAR(Real, SamplingDistance, 0.005, "Emitter Sampling Distance");

		DArray<Coord> gen_pos;
		DArray<Coord> gen_vel;

		DArray<Coord> pos_buf;
		DArray<Coord> vel_buf;
		DArray<Coord> force_buf;
		int sum = 0;
	private:
		
	};
}