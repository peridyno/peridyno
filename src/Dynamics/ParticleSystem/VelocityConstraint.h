#pragma once
#include "Module/ConstraintModule.h"
#include "Algorithm/Reduction.h"
#include "Algorithm/Functional.h"
#include "Algorithm/Arithmetic.h"
#include "Algorithm/Reduction.h"

namespace dyno {

	class Attribute;
	template<typename TDataType> class SummationDensity;

	template<typename TDataType>
	class VelocityConstraint : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		VelocityConstraint();
		~VelocityConstraint() override;
		
		void constrain() override;

	public:
		DEF_VAR(Real, RestDensity, Real(1000), "");

		DEF_VAR_IN(Real, TimeStep, "Time step size");

		DEF_VAR_IN(Real, SamplingDistance, "");

		DEF_VAR_IN(Real, SmoothingLength, "");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

		//FVar<Real> m_smoothingLength;
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");

		DeviceArrayField<Coord> m_normal;
		DeviceArrayField<Attribute> m_attribute;
		
		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "");


	protected:
		bool initializeImpl() override;

	private:
		bool m_bConfigured = false;
		Real m_maxAlpha;
		Real m_maxA;
		Real m_airPressure = 0.0f;

		Real m_particleMass = 1.0f;
		Real m_tangential = 0.1f;
		Real m_separation = 0.1f;
		Real m_restDensity = 1000.0f;

		//Refer to "A Nonlocal Variational Particle Framework for Incompressible Free Surface Flows" for their exact meanings
		DArray<Real> m_alpha;
		DArray<Real> m_Aii;
		DArray<Real> m_AiiFluid;
		DArray<Real> m_AiiTotal;

		DArray<Real> m_pressure;
		DArray<Real> m_divergence;
		//Indicate whether a particle is near the free surface boundary.
		DArray<bool> m_bSurface;

		//Used to solve the linear system of equations with a conjugate gradient method.
		DArray<Real> m_y;
		DArray<Real> m_r;
		DArray<Real> m_p;

		Reduction<Real>* m_reduce;
		Arithmetic<Real>* m_arithmetic;

		std::shared_ptr<SummationDensity<TDataType>> m_densitySum;
	};
}