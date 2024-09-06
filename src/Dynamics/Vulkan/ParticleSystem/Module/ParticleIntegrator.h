#pragma once
#include "Module/NumericalIntegrator.h"

namespace dyno 
{
	class ParticleIntegrator : public Module
	{
		DECLARE_CLASS(ParticleIntegrator)

	public:
		ParticleIntegrator();
		~ParticleIntegrator() override {};
		
	public:

		DEF_VAR_IN(float, TimeStep, "Time step size");

		/**
		* @brief Position
		* Particle position
		*/
		DEF_ARRAY_IN(Vec3f, Position, DeviceType::GPU, "Particle position");

		/**
		* @brief Velocity
		* Particle velocity
		*/
		DEF_ARRAY_IN(Vec3f, Velocity, DeviceType::GPU, "Particle velocity");

	protected:
		void updateImpl() override;

	
	};
}