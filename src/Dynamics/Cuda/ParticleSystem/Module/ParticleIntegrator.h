#pragma once
#include "Module/NumericalIntegrator.h"

#include "../Attribute.h"

namespace dyno {
	template<typename TDataType>
	class ParticleIntegrator : public NumericalIntegrator
	{
		DECLARE_TCLASS(ParticleIntegrator, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleIntegrator();
		~ParticleIntegrator() override {};
		
		void begin() override;
		void end() override;

		bool integrate() override;

		bool updateVelocity();
		bool updatePosition();

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

		/**
		* @brief Force density
		* Force density on each particle
		*/
		DEF_ARRAY_IN(Coord, ForceDensity, DeviceType::GPU, "Force density on each particle");

	protected:
		void updateImpl() override;

	private:
		DArray<Coord> m_prePosition;
		DArray<Coord> m_preVelocity;
		int cnt = 0;
	};

	IMPLEMENT_TCLASS(ParticleIntegrator, TDataType)
}