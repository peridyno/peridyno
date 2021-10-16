#pragma once
#include "Node.h"
#include "RigidBodyShared.h"

#include "Topology/Primitive3D.h"
#include "Collision/NeighborElementQuery.h"

namespace dyno
{
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

		typedef typename TContactPair<Real> ContactPair;

		RigidBodySystem(std::string name = "RigidBodySystem");
		virtual ~RigidBodySystem();

		void addBox(
			const BoxInfo& box, 
			const RigidBodyInfo& bodyDef,
			const Real density = Real(1));

		void addSphere(
			const SphereInfo& sphere,
			const RigidBodyInfo& bodyDef, 
			const Real density = Real(1));

		void addTet(
			const TetInfo& tet,
			const RigidBodyInfo& bodyDef,
			const Real density = Real(1));

	public:
		void setUpperCorner(Coord h) { mUpperCorner = h; }
		void setLowerCorner(Coord l) { mLowerCorner = l; }

	protected:
		void resetStates() override;
		void updateStates() override;

		void updateTopology() override;

	private:
		void initializeJacobian(Real dt);
		void detectCollisionWithBoundary();

	public:
		DEF_VAR(bool, FrictionEnabled, true, "A toggle to control the friction");

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
		DEF_EMPTY_CURRENT_ARRAY(RotationMatrix, Matrix, DeviceType::GPU, "Rotation matrix of rigid bodies");

		DEF_EMPTY_CURRENT_ARRAY(Inertia, Matrix, DeviceType::GPU, "Interial matrix");

		DEF_EMPTY_CURRENT_ARRAY(Quaternion, TQuat, DeviceType::GPU, "Quaternion");

		DEF_ARRAY_STATE(CollisionMask, CollisionMask, DeviceType::GPU, "Collision mask for each rigid body");


		std::shared_ptr<NeighborElementQuery<TDataType>> mElementQuery;

	private:
		std::vector<RigidBodyInfo> mHostRigidBodyStates;

		std::vector<SphereInfo> mHostSpheres;
		std::vector<BoxInfo> mHostBoxes;
		std::vector<TetInfo> mHostTets;

		DArray<RigidBodyInfo> mDeviceRigidBodyStates;

		DArray<SphereInfo> mDeviceSpheres;
		DArray<BoxInfo> mDeviceBoxes;
		DArray<TetInfo> mDeviceTets;

		DArray<Matrix> mInitialInertia;

		DArray<Coord> mJ;		//Jacobian
		DArray<Coord> mB;		//B = M^{-1}J^T
		DArray<Coord> mAccel;

		DArray<Real> mEta;		//eta
		DArray<Real> mD;		//diagonal elements of JB
		DArray<Real> mLambda;	//contact impulse

		DArray<Real> nbrContacts;

		DArray<int> mBoundaryContactCounter;
		DArray<ContactPair> mBoundaryContacts;
		DArray<ContactPair> buffer_friction;
		DArray<ContactPair> mAllConstraints;

	private:
		//TODO: add collision support with triangular mesh
		bool have_mesh = false;
		bool have_mesh_boundary = false;

		Reduction<int> m_reduce;
		Scan m_scan;

		Coord mUpperCorner = Coord(100, 5, 100);//(0.4925,0.4925,0.4925);
		Coord mLowerCorner = Coord(-100, 0, -100);//(0.0075,0.0075,0.0075);
	};
}