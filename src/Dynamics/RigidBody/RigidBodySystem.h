#pragma once
#include "Node.h"
#include "RigidBodyShared.h"
//#include "Core/Quaternion/quaternion.h"
#include "Topology/Primitive3D.h"
#include "Topology/NeighborElementQuery.h"
#include "Topology/DiscreteElements.h"
#include "Matrix.h"
#include "Topology/NeighborConstraints.h"
#include "Quat.h"

namespace dyno
{
	typedef typename TOrientedBox3D<Real> Box3D;

	#define constraint_friction 5
	#define	constraint_boundary 0
	#define constraint_collision 1
	#define	constraint_distance -2
	template<typename TDataType> class DiscreteElements;
	typedef typename TNeighborConstraints<Real> NeighborConstraints;
	/*!
	*	\class	RigidBodySystem
	*	\brief	Implementation of a rigid body system containing a variety of rigid bodies with different shapes.
	*
	*/
	template<typename TDataType>
	class RigidBodySystem : public Node
	{
		DECLARE_CLASS_1(RigidBodySystem, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename TSphere3D<Real> Sphere3D;
		typedef typename TOrientedBox3D<Real> Box3D;
		typedef typename Quat<Real> TQuat;


		RigidBodySystem(std::string name = "RigidBodySystem");
		virtual ~RigidBodySystem();

		void addBox(
			const RigidBodyInfo& bodyDef, 
			const BoxInfo& box, 
			const Real density = Real(1));

		void addSphere(
			const RigidBodyInfo& bodyDef, 
			const SphereInfo& sphere, 
			const Real density = Real(1));
		
		void solve_constraint();
		void update_position_rotation(Real dt);
		void init_jacobi(Real dt);
		void init_boundary();
		void init_friction();

		void pretreat(Real dt);
		void take_one_iteration(Real dt);
		void update_state(Real dt);
		
		

		
	public:
		void set_hi(Coord h) { hi = h; }
		void set_lo(Coord l) { lo = l; }

		DeviceArrayField<Coord> AA;

		bool late_initialize = false;
		int size_else;

		std::vector<Coord> host_angular_velocity;
		std::vector<Coord> host_velocity;
		std::vector<Matrix> host_inertia_tensor;
		std::vector<Real> host_mass;
		std::vector<Coord> host_pair_point;

		std::vector<RigidBodyInfo> mHostRigidBodyStates;
		
		std::vector<SphereInfo> mSpheres;
		std::vector<BoxInfo> mBoxes;

	protected:
		void resetStates() override;
		void updateStates() override;

	private:
		/**
		 * @brief Particle position
		 */
		DEF_EMPTY_CURRENT_ARRAY(Mass, Real, DeviceType::GPU, "Mass of rigid bodies");


		/**
		 * @brief Particle position
		 */
		DEF_EMPTY_CURRENT_ARRAY(Center, Coord, DeviceType::GPU, "Center of rigid bodies");


		/**
		 * @brief Particle position
		 */
		DEF_EMPTY_CURRENT_ARRAY(Velocity, Coord, DeviceType::GPU, "Velocity of rigid bodies");

		/**
		 * @brief Particle position
		 */
		DEF_EMPTY_CURRENT_ARRAY(AngularVelocity, Coord, DeviceType::GPU, "Angular velocity of rigid bodies");


		/**
		 * @brief Particle position
		 */
		DEF_EMPTY_CURRENT_ARRAY(RigidRotation, Matrix, DeviceType::GPU, "Rotation matrix of rigid bodies");

		DeviceArrayField<Matrix> m_inertia;
		DArray<Matrix> m_inertia_init;
		DArray<Sphere3D> m_sphere3d_init;
		DArray<Box3D> m_box3d_init;
		DArray<Tet3D> m_tet3d_init;
		DArray<TQuat> m_rotation_q;

		DArray<int> m_Constraints;



		DArray<Coord> pos_init;
		DArray<Coord> center_init;
		DArray<Coord> pair_point;
		DArray<int> pair;


		DArray<Coord> J;
		DArray<Coord> B;
		DArray<Real> ita;
		DArray<Real> D;

		DArray<Real> lambda;

		DArray<Real> mass_eq;
		DArray<Real> mass_buffer;

		bool use_new_mass = false;//true;

		DArray<int> cnt_boudary;
		DArray<NeighborConstraints> buffer_boundary;
		DArray<NeighborConstraints> buffer_friction;
		DArray<NeighborConstraints> constraints_all;

		void rigid_update_topology();
	private:
		std::shared_ptr<NeighborElementQuery<TDataType>>m_nbrQueryElement;


		bool have_mesh = false;
		bool have_friction = true;
		bool have_mesh_boundary = false;

		int start_box;
		int start_sphere;
		int start_tet;
		int start_segment;

		int size_mesh;

		Reduction<int> m_reduce;
		Scan m_scan;

		Coord hi = Coord(100,5,100);//(0.4925,0.4925,0.4925);
		Coord lo = Coord(-100, 0, -100);//(0.0075,0.0075,0.0075);
	};



//#ifdef PRECISION_FLOAT
//	template class RigidBodySystem<DataType3f>;
//#else
//	template class RigidBodySystem<DataType3d>;
//#endif
}