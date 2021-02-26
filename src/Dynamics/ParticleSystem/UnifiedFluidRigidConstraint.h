#pragma once
#include "Framework/ModuleConstraint.h"
#include "Framework/FieldArray.h"
#include "Utility.h"
#include "Framework/FieldVar.h"
#include "Topology/FieldNeighbor.h"
#include "Framework/ModuleTopology.h"
//#include "Topology/Primitive3D.h"
#include "RigidBody/RigidBodySystem.h"

namespace dyno {

	class Attribute;
	template <typename TDataType> class SummationDensity;
	template <typename TDataType> class DensitySummationMesh;
	template <typename TDataType> class TriangleSet;
	


	template<typename TDataType>
	class UnifiedFluidRigidConstraint : public Node
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename TDataType::Matrix Matrix;
		typedef typename Quaternion<Real> TQuaternion;

		UnifiedFluidRigidConstraint();
		~UnifiedFluidRigidConstraint() override;

		void pretreat(Real dt);
		void take_one_iteration_1(Real dt);
		void take_one_iteration_2(Real dt);
		void update(Real dt);


		//void update_state_rigid();
		//void calculate_gradient();


		void setDiscreteSet(std::shared_ptr<DiscreteElements<TDataType>> d)
		{
			m_shapes = d;
		}

	public:
		DeviceArrayField<Coord> m_particle_position;//
		DeviceArrayField<Coord> m_particle_velocity;
		DeviceArrayField<Coord> AA; // AA in rigids, used for : a) update rigid velocity state   b) updated by fluid
		

		DeviceArrayField<Coord> m_rigid_velocity;//
		DeviceArrayField<Coord> m_rigid_angular_velocity;//
		DeviceArrayField<Coord> m_rigid_position;//
		DeviceArrayField<Matrix> m_rigid_rotation;//
		//DeviceArrayField<TQuaternion> m_rigid_rotaion_q;

		
		DeviceArrayField<Coord> m_fluid_tmp_vel;

		DeviceArrayField<Real> m_gradient_point;
		DeviceArrayField<Real> m_gradient_boundary;
		DeviceArrayField<Real> m_boundary_forces;


		DeviceArrayField<Real> m_rigid_mass;
		DeviceArrayField<Matrix> m_rigid_interior;

		DeviceArrayField<NeighborConstraints> m_nbr_cons;

	//protected:
		bool initialize() override;

	private:

		Arithmetic<Real>* m_arithmetic = NULL;
		std::shared_ptr<DiscreteElements<TDataType>> m_shapes;
		std::shared_ptr<NeighborElementQuery<TDataType>>m_nbrQueryElement;
		
		GArray<Coord> tmp_rigid_velocity;
		GArray<Coord> tmp_rigid_angular_velocity;

		GArray<Real> delta_force;

		Real sampling_distance = 0.005f;
		Real restDensity = 1000.0f;


		Real err_last;
	};



#ifdef PRECISION_FLOAT
	template class UnifiedFluidRigidConstraint<DataType3f>;
#else
	template class UnifiedFluidRigidConstraint<DataType3d>;
#endif
}