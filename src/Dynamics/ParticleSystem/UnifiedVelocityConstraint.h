#pragma once
#include "Framework/ModuleConstraint.h"
#include "Framework/FieldArray.h"
#include "Utility.h"
#include "Framework/FieldVar.h"
#include "Topology/FieldNeighbor.h"
#include "Framework/ModuleTopology.h"
//#include "Topology/Primitive3D.h"
#include "UnifiedFluidRigidConstraint.h"

namespace dyno {

	class Attribute;
	template <typename TDataType> class SummationDensity;
	template <typename TDataType> class DensitySummationMesh;
	template <typename TDataType> class TriangleSet;
	//template <typename TDataType> class Point3D;
	//template <typename TDataType> class Triangle3D;
	//template <typename TDataType> class Plane3D;


	template<typename TDataType>
	class UnifiedVelocityConstraint : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		UnifiedVelocityConstraint();
		~UnifiedVelocityConstraint() override;

		bool constrain() override;

		void pretreat(Real dt);
		void take_one_iteration_1(Real dt);
		void take_one_iteration_2(Real dt);
		void take_one_iteration(Real dt);
		void update(Real dt);
		void update1(Real dt);
		//	DeviceArrayField<Coord>* getPosition() override;
		DeviceArrayField<Coord>* getPosition()
		{
			return &m_particle_position;
		}//override;

	public:
		

		DeviceArrayField<Real> m_pressure;//
		DeviceArrayField<Real> m_gradient_point;//
		DeviceArrayField<NeighborConstraints> m_nbrcons;//

		DeviceArrayField<Real> m_gradient_rigid;//
		DeviceArrayField<Real> m_force_rigid;//
		DeviceArrayField<Coord> m_velocity_inside_iteration;//

		VarField<Real> m_smoothing_length;
		VarField<Real> m_sampling_distance;

		/**
		 * @brief Particle attributes
		 *
		 */
		DeviceArrayField<Real> m_particle_mass;

		DeviceArrayField<Coord> m_particle_position;
		DeviceArrayField<Coord> m_particle_velocity;
		GArray<Coord> particle_velocity_buffer;
		//		DeviceArrayField<Coord> m_particle_normal;

		DeviceArrayField<Attribute> m_particle_attribute;
		DeviceArrayField<int> m_flip;

		/**
		 * @brief Solid wall boundary
		 *
		 */
		DeviceArrayField<Real> m_triangle_vertex_mass;
		DeviceArrayField<Coord> m_triangle_vertex;
		DeviceArrayField<Coord> m_triangle_vertex_old;
		DeviceArrayField<Triangle> m_triangle_index;


		DeviceArrayField<Real> m_pressure_point;

		

		/**
		 * @brief Storing neighboring particles and triangles' ids
		 *
		 */
		NeighborField<int> m_neighborhood_particles;
		NeighborField<int> m_neighborhood_triangles;

	protected:
		bool initializeImpl() override;

	private:
		bool m_bConfigured = false;
		Real m_maxAlpha;
		Real m_maxA;
		Real m_airPressure = 10000.0f;

		Real m_particleMass = 1.0f;
		Real m_tangential = 0.0f;
		Real m_separation = 0.0f;
		Real m_restDensity = 1000.0f;

		Real err, err_last;

		int num_f;
		int start_f = 0;
		bool first_step = false;


		//Refer to "A Nonlocal Variational Particle Framework for Incompressible Free Surface Flows" for their exact meanings
		GArray<Real> m_alpha;
		GArray<Real> Rho_alpha;
		GArray<Real> m_Aii;
		GArray<Real> m_AiiFluid;
		GArray<Real> m_AiiTotal;

		//DeviceArrayField<Real> m_density_field;
		GArray<Real> m_density;
		DeviceArrayField<Real> m_density_field;

		GArray<Real> invRadius;


		

		
		GArray<Real> m_divergence;
		GArray<Real> m_gradient;
		GArray<Real> m_step;
		//Indicate whether a particle is near the free surface boundary.
		GArray<bool> m_bSurface;

		//Used to solve the linear system of equations with a conjugate gradient method.
		GArray<Real> m_y;
		GArray<Real> m_r;
		GArray<Real> m_p;

		Reduction<Real>* m_reduce;
		Arithmetic<Real>* m_arithmetic;

		GArray<Coord> m_meshVel;

		GArray<Real> m_pairwise_force;
		GArray<int> m_mapping;
		GArray<int> m_index_sym;

		GArray<Coord> m_particle_velocity_buffer;
		

		std::shared_ptr<DensitySummationMesh<TDataType>> m_densitySum;
	//	std::shared_ptr<SummationDensity<TDataType>> m_densitySum;

		Real step_i;

		
	};



#ifdef PRECISION_FLOAT
	template class UnifiedVelocityConstraint<DataType3f>;
#else
	template class UnifiedVelocityConstraint<DataType3d>;
#endif
}