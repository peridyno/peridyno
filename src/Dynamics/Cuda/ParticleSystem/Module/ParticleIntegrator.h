#pragma once
#include "Module/ComputeModule.h"

#include "Collision/Attribute.h"

namespace dyno {
	template<typename TDataType>
	class ParticleIntegrator : public ComputeModule
	{
		DECLARE_TCLASS(ParticleIntegrator, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleIntegrator();
		~ParticleIntegrator() override {};
		
	public:

		DEF_VAR_IN(Real, TimeStep, "Time step size");

		/**
		* @brief Position
		* Particle position
		*/
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Particle position");

		/**
		* @brief Velocity
		* Particle velocity
		*/
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Particle velocity");

		/**
		* @brief Attribute
		* Particle attribute
		*/
		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "Particle attribute");

	protected:
		void compute() override;

	private:
		bool updateVelocity();
		bool updatePosition();
	};
}