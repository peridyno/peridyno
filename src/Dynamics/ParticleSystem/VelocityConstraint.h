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
		GArray<Real> m_alpha;
		GArray<Real> m_Aii;
		GArray<Real> m_AiiFluid;
		GArray<Real> m_AiiTotal;

		GArray<Real> m_density;

		GArray<Real> m_pressure;
		GArray<Real> m_divergence;
		//Indicate whether a particle is near the free surface boundary.
		GArray<bool> m_bSurface;

		//Used to solve the linear system of equations with a conjugate gradient method.
		GArray<Real> m_y;
		GArray<Real> m_r;
		GArray<Real> m_p;

		Reduction<Real>* m_reduce;
		Arithmetic<Real>* m_arithmetic;

		std::shared_ptr<SummationDensity<TDataType>> m_densitySum;
	};



#ifdef PRECISION_FLOAT
	template class VelocityConstraint<DataType3f>;
#else
	template class VelocityConstraint<DataType3d>;
#endif
}