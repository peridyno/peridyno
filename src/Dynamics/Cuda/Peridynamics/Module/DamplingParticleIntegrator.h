#pragma once
#include "Module/NumericalIntegrator.h"

#include "Collision/Attribute.h"

namespace dyno {
	template<typename TDataType>
	class DamplingParticleIntegrator : public NumericalIntegrator
	{
		DECLARE_TCLASS(DamplingParticleIntegrator, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DamplingParticleIntegrator();
		~DamplingParticleIntegrator() override {};
		
		void begin() override;
		void end() override;

		bool integrate() override;

		bool updateVelocity() ;
		bool updatePosition() ;

	public:

		DEF_ARRAY_IN(Coord, ContactForce, DeviceType::GPU, "Contact force")

		DEF_ARRAY_IN(Coord, DynamicForce, DeviceType::GPU, "Contact force");

		DEF_ARRAY_IN(Coord, Norm, DeviceType::GPU, "vertex norm");

		DEF_VAR_IN(Real, Mu, "friction parameter");

		DEF_VAR_IN(Real, AirDisspation, " air disspation");

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

		/**
		* @brief Force density
		* Force density on each particle
		*/
		DEF_ARRAY_IN(Coord, ForceDensity, DeviceType::GPU, "Force density on each particle");

	protected:
		void updateImpl() override;

	};

	IMPLEMENT_TCLASS(DamplingParticleIntegrator, TDataType)
}