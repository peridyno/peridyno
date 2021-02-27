#pragma once
#include "Framework/Node.h"
#include "Topology/Primitive3D.h"
#include "Topology/NeighborElementQuery.h"

namespace dyno
{
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
		typedef typename Quat<Real> TQuaternion;


		RigidBodySystem(std::string name = "RigidBodySystem");
		virtual ~RigidBodySystem();

		void advance(Real dt) override;
		void solve_constraint();
		void update_position_rotation(Real dt);
		void init_jacobi(Real dt);
		void init_boundary();

		void pretreat(Real dt);
		void take_one_iteration(Real dt);
		void update(Real dt);
		

	public:
		bool initialize() override;
		void set_hi(Coord h) { hi = h; }
		void set_lo(Coord l) { lo = l; }

		DeviceArrayField<Coord> AA;

		bool late_initialize = false;

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
		DEF_EMPTY_CURRENT_ARRAY(Rotation, Matrix, DeviceType::GPU, "Rotation matrix of rigid bodies");

		DeviceArrayField<Matrix> m_inertia;
		GArray<Matrix> m_inertia_init;
		GArray<Sphere3D> m_sphere3d_init;
		GArray<Box3D> m_box3d_init;
		GArray<TQuaternion> m_rotation_q;

		GArray<int> m_Constraints;



		GArray<Coord> pos_init;
		GArray<Coord> center_init;
		GArray<Coord> pair_point;
		GArray<int> pair;


		
		GArray<Coord> J;
		GArray<Coord> B;
		GArray<Real> ita;
		GArray<Real> D;
		GArray<Real> lambda;



		GArray<int> cnt_boudary;
		GArray<NeighborConstraints> buffer_boundary;
		GArray<NeighborConstraints> constraints_all;

		void rigid_update_topology();
	private:
		std::shared_ptr<DiscreteElements<TDataType>> m_shapes;
		std::shared_ptr<NeighborElementQuery<TDataType>>m_nbrQueryElement;

		int start_box;
		int start_sphere;

		Reduction<int> m_reduce;
		Scan m_scan;

		Coord hi = Coord(0.4925,0.4925,0.4925);
		Coord lo = Coord(0.0075,0.0075,0.0075);
	};



#ifdef PRECISION_FLOAT
	template class RigidBodySystem<DataType3f>;
#else
	template class RigidBodySystem<DataType3d>;
#endif
}