#pragma once
#include "Framework/NumericalModel.h"
#include "Framework/FieldVar.h"
#include "Framework/FieldArray.h"
#include "DensityPBD.h"
#include "UnifiedVelocityConstraint.h"
#include "Attribute.h"
#include "Framework/ModuleTopology.h"
#include "MeshCollision.h"

namespace dyno
{
	template<typename TDataType> class PointSetToPointSet;
	template<typename TDataType> class ParticleIntegrator;
	template<typename TDataType> class NeighborQuery;
	template<typename TDataType> class NeighborTriangleQuery;
	template<typename TDataType> class DensityPBD;
	template<typename TDataType> class MeshCollision;
	template<typename TDataType> class SurfaceTension;
	template<typename TDataType> class ImplicitViscosity;
	template<typename TDataType> class Helmholtz;
	template<typename> class PointSetToPointSet;
	typedef typename TopologyModule::Triangle Triangle;

	class ForceModule;
	class ConstraintModule;
	class Attribute;
	/*!
	*	\class	ParticleSystem
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class UnifiedVelocityNode : public NumericalModel
	{
		DECLARE_CLASS_1(UnifiedVelocityNode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		UnifiedVelocityNode();
		virtual ~UnifiedVelocityNode();

		void step(Real dt) override;

		void setSmoothingLength(Real len) { m_smoothing_length.setValue(len); }
		void setRestDensity(Real rho) { m_restRho = rho; }

		void setIncompressibilitySolver(std::shared_ptr<ConstraintModule> solver);
		void setViscositySolver(std::shared_ptr<ConstraintModule> solver);
		void setSurfaceTensionSolver(std::shared_ptr<ConstraintModule> solver);

		void pretreat(Real dt);
		void take_one_iteration1(Real dt);
		void take_one_iteration2(Real dt);
		void update(Real dt);

		DeviceArrayField<Real>* getDensityField()
		{
			return m_pbdModule2->outDensity();
			//return m_force_density;
		}

	public:
		bool late_initialize = false;

		VarField<Real> m_smoothing_length;

		VarField<Real> max_vel;
		VarField<Real> var_smoothing_length;
		
		DeviceArrayField<Real> m_particle_mass;

		DeviceArrayField<Coord> m_particle_position;
		DeviceArrayField<Coord> m_particle_velocity;

		DeviceArrayField<Attribute> m_particle_attribute;


		DeviceArrayField<Real> m_triangle_vertex_mass;
		DeviceArrayField<Coord> m_triangle_vertex;
		DeviceArrayField<Coord> m_triangle_vertex_old;
		DeviceArrayField<Triangle> m_triangle_index;

		DeviceArrayField<Coord> m_particle_force_density;
		DeviceArrayField<Coord> m_vertex_force_density;
		DeviceArrayField<Coord> m_vn;


		DeviceArrayField<Real> m_pressure;//
		DeviceArrayField<Real> m_gradient_point;//
		DeviceArrayField<NeighborConstraints> m_nbrcons;//
		DeviceArrayField<Real> m_gradient_rigid;//
		DeviceArrayField<Real> m_force_rigid;//
		DeviceArrayField<Coord> m_velocity_inside_iteration;//


		DeviceArrayField<int> m_flip;
		Reduction<Real>* pReduce;

		

		DeviceArrayField<Coord> m_velocity_mod;


	protected:
		bool initializeImpl() override;

	private:
		int m_pNum;
		Real m_restRho;
		int first = 1;

		//std::shared_ptr<ConstraintModule> m_surfaceTensionSolver;
		std::shared_ptr<ConstraintModule> m_viscositySolver;
		
		std::shared_ptr<ConstraintModule> m_incompressibilitySolver;

		std::shared_ptr<UnifiedVelocityConstraint<TDataType>> m_pbdModule;


		std::shared_ptr<DensityPBD<TDataType>> m_pbdModule2;

		std::shared_ptr<MeshCollision<TDataType>> m_meshCollision;

		std::shared_ptr<ImplicitViscosity<TDataType>> m_visModule;
		std::shared_ptr<SurfaceTension<TDataType>>  m_surfaceTensionSolver;
		std::shared_ptr<Helmholtz<TDataType>> m_Helmholtz;
		std::shared_ptr<PointSetToPointSet<TDataType>> m_mapping;
		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<NeighborQuery<TDataType>>m_nbrQueryPoint;
		
		std::shared_ptr<NeighborTriangleQuery<TDataType>>m_nbrQueryTri;
		std::shared_ptr<NeighborQuery<TDataType>>m_nbrQueryTriMulti;
	};

#ifdef PRECISION_FLOAT
	template class UnifiedVelocityNode<DataType3f>;
#else
	template class UnifiedVelocityNode<DataType3d>;
#endif
}