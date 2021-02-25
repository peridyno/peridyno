#pragma once
#include "Framework/ModuleConstraint.h"
#include "Framework/FieldArray.h"
#include "Utility.h"
#include "Framework/FieldVar.h"
#include "Topology/FieldNeighbor.h"
#include "Framework/ModuleTopology.h"
//#include "Topology/Primitive3D.h"

namespace dyno {

	class Attribute;
	template <typename TDataType> class SummationDensity;
	template <typename TDataType> class TriangleSet;
	//template <typename TDataType> class Point3D;
	//template <typename TDataType> class Triangle3D;
	//template <typename TDataType> class Plane3D;


	template<typename TDataType>
	class SemiAnalyticalIncompressibilityModule : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		SemiAnalyticalIncompressibilityModule();
		~SemiAnalyticalIncompressibilityModule() override;
		
		bool constrain() override;
	//	DeviceArrayField<Coord>* getPosition() override;
		DeviceArrayField<Coord>* getPosition()
		{
			return &m_particle_position;
		}//override;

	public:
		VarField<Real> m_smoothing_length;
		VarField<Real> m_sampling_distance;

		/**
		 * @brief Particle attributes
		 * 
		 */
		DeviceArrayField<Real> m_particle_mass;

		DeviceArrayField<Coord> m_particle_position;
		DeviceArrayField<Coord> m_particle_velocity;
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
		Real m_tangential = 1.0f;
		Real m_separation = 1.0f;
		Real m_restDensity = 1000.0f;

		int num_f;
		int start_f = 0;
		bool first_step = false;
		

		//Refer to "A Nonlocal Variational Particle Framework for Incompressible Free Surface Flows" for their exact meanings
		DeviceArray<Real> m_alpha;
		DeviceArray<Real> Rho_alpha;
		DeviceArray<Real> m_Aii;
		DeviceArray<Real> m_AiiFluid;
		DeviceArray<Real> m_AiiTotal;

		//DeviceArrayField<Real> m_density_field;
		DeviceArray<Real> m_density;

		DeviceArray<Real> m_pressure;
		DeviceArray<Real> m_divergence;
		//Indicate whether a particle is near the free surface boundary.
		DeviceArray<bool> m_bSurface;

		//Used to solve the linear system of equations with a conjugate gradient method.
		DeviceArray<Real> m_y;
		DeviceArray<Real> m_r;
		DeviceArray<Real> m_p;

		Reduction<Real>* m_reduce;
		Arithmetic<Real>* m_arithmetic;

		DeviceArray<Coord> m_meshVel;

		std::shared_ptr<SummationDensity<TDataType>> m_densitySum;
	};



#ifdef PRECISION_FLOAT
	template class SemiAnalyticalIncompressibilityModule<DataType3f>;
#else
	template class SemiAnalyticalIncompressibilityModule<DataType3d>;
#endif
}