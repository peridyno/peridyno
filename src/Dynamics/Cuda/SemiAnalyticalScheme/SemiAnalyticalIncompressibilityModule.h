/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-08-06
 * @description: Declaration of SemiAnalyticalIncompressibilityModule class, implemendation of semi-analytical perojection-based fluids 
 *               introduced in the paper <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * @version    : 1.1
 */
#pragma once
#include "Module/ConstraintModule.h"
#include "Module/TopologyModule.h"
#include "Module/TopologyMapping.h"
#include "Algorithm/Arithmetic.h"

namespace dyno 
{
	class Attribute;
	template <typename TDataType>
	class SummationDensity;
	template <typename TDataType>
	class TriangleSet;
	/**
	 * SemiAnalyticalIncompressibilityModule implements the projection-based part of semi-analytical boundary conditions of the paper
	 * <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
	 * It is used in SemiAnalyticalIncompressibleFluidModel class
	 */

	template <typename TDataType>
	class SemiAnalyticalIncompressibilityModule : public ConstraintModule
	{
	public:
		typedef typename TDataType::Real          Real;
		typedef typename TDataType::Coord         Coord;
		typedef typename TopologyModule::Triangle Triangle;

		SemiAnalyticalIncompressibilityModule();
		~SemiAnalyticalIncompressibilityModule() override;

		/**
		 * enforce projection-based fluids with semi-analytical boundary conditions
		 *
		 * @return(always)
		 */
		void constrain() override;

		DeviceArrayField<Coord>* getPosition()
		{
			return &m_particle_position;
		}  //override;

	public:
		FVar<Real> m_smoothing_length;
		FVar<Real> m_sampling_distance;

		DeviceArrayField<Real> m_particle_mass;

		DeviceArrayField<Coord> m_particle_position;
		DeviceArrayField<Coord> m_particle_velocity;

		DeviceArrayField<Attribute> m_particle_attribute;
		DeviceArrayField<int>       m_flip;

		DeviceArrayField<Real>     m_triangle_vertex_mass;
		DeviceArrayField<Coord>    m_triangle_vertex;
		DeviceArrayField<Coord>    m_triangle_vertex_old;
		DeviceArrayField<Triangle> m_triangle_index;

		/**
			 * @brief Storing neighboring particles and triangles' ids
			 *
			 */
		DEF_ARRAYLIST_IN(int, NeighborParticleIds, DeviceType::GPU, "");// m_neighborhood_particles;
		DEF_ARRAYLIST_IN(int, NeighborTriangleIds, DeviceType::GPU, "");

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

		int  num_f;
		int  start_f = 0;
		bool first_step = false;

		//Refer to "A Nonlocal Variational Particle Framework for Incompressible Free Surface Flows" for their exact meanings
		DArray<Real> m_alpha;
		DArray<Real> Rho_alpha;
		DArray<Real> m_Aii;
		DArray<Real> m_AiiFluid;
		DArray<Real> m_AiiTotal;

		//DeviceArrayField<Real> m_density_field;
		DArray<Real> m_density;

		DArray<Real> m_pressure;
		DArray<Real> m_divergence;
		//Indicate whether a particle is near the free surface boundary.
		DArray<bool> m_bSurface;

		//Used to solve the linear system of equations with a conjugate gradient method.
		DArray<Real> m_y;
		DArray<Real> m_r;
		DArray<Real> m_p;

		Reduction<Real>*  m_reduce;
		Arithmetic<Real>* m_arithmetic;

		DArray<Coord> m_meshVel;

		std::shared_ptr<SummationDensity<TDataType>> m_densitySum;
	};
}  // namespace dyno