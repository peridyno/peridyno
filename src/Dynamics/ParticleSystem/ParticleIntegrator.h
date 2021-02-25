#pragma once
#include "Framework/NumericalIntegrator.h"
#include "Framework/FieldVar.h"
#include "Framework/FieldArray.h"
#include "Attribute.h"

namespace dyno {
	template<typename TDataType>
	class ParticleIntegrator : public NumericalIntegrator
	{
		DECLARE_CLASS_1(ParticleIntegrator, TDataType)

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

	protected:
		bool initializeImpl() override;

	public:

		/**
		* @brief Position
		* Particle position
		*/
		DEF_EMPTY_IN_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");

		/**
		* @brief Velocity
		* Particle velocity
		*/
		DEF_EMPTY_IN_ARRAY(Velocity, Coord, DeviceType::GPU, "Particle velocity");

		/**
		* @brief Attribute
		* Particle attribute
		*/
		DEF_EMPTY_IN_ARRAY(Attribute, Attribute, DeviceType::GPU, "Particle attribute");

		/**
		* @brief Force density
		* Force density on each particle
		*/
		DEF_EMPTY_IN_ARRAY(ForceDensity, Coord, DeviceType::GPU, "Force density on each particle");


	private:
		DeviceArray<Coord> m_prePosition;
		DeviceArray<Coord> m_preVelocity;
	};

#ifdef PRECISION_FLOAT
	template class ParticleIntegrator<DataType3f>;
#else
 	template class ParticleIntegrator<DataType3d>;
#endif
}