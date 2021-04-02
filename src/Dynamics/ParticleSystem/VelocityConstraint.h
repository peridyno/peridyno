#pragma once
#include "Framework/ModuleConstraint.h"
#include "Topology/FieldNeighbor.h"
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
		
		bool constrain() override;

	public:
		VarField<Real> m_smoothingLength;

		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_normal;
		DeviceArrayField<Attribute> m_attribute;
		NeighborField<int> m_neighborhood;

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

		DArray<Real> m_density;

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