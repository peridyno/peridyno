#include "RigidBodySystem.h"

#include "Primitive/Primitive3D.h"
#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"
#include <cuda_runtime.h>
#include "IterativeConstraintSolver.h"

//Module headers
#include "ContactsUnion.h"


namespace dyno
{
	IMPLEMENT_TCLASS(RigidBodySystem, TDataType)

	typedef typename dyno::TOrientedBox3D<Real> Box3D;

	template<typename TDataType>
	RigidBodySystem<TDataType>::RigidBodySystem()
		: Node()
	{
		auto defaultTopo = std::make_shared<DiscreteElements<TDataType>>();
		this->stateTopology()->setDataPtr(std::make_shared<DiscreteElements<TDataType>>());

		auto elementQuery = std::make_shared<NeighborElementQuery<TDataType>>();
		this->stateTopology()->connect(elementQuery->inDiscreteElements());
		this->stateCollisionMask()->connect(elementQuery->inCollisionMask());
		this->animationPipeline()->pushModule(elementQuery);

		auto cdBV = std::make_shared<CollistionDetectionBoundingBox<TDataType>>();
		this->stateTopology()->connect(cdBV->inDiscreteElements());
		this->animationPipeline()->pushModule(cdBV);

		auto merge = std::make_shared<ContactsUnion<TDataType>>();
		elementQuery->outContacts()->connect(merge->inContactsA());
		cdBV->outContacts()->connect(merge->inContactsB());
		this->animationPipeline()->pushModule(merge);

		auto iterSolver = std::make_shared<IterativeConstraintSolver<TDataType>>();
		this->stateTimeStep()->connect(iterSolver->inTimeStep());
		this->varFrictionEnabled()->connect(iterSolver->varFrictionEnabled());
		this->varGravityEnabled()->connect(iterSolver->varGravityEnabled());
		this->varGravityValue()->connect(iterSolver->varGravityValue());
		this->varFrictionCoefficient()->connect(iterSolver->varFrictionCoefficient());
		this->varSlop()->connect(iterSolver->varSlop());
		this->stateMass()->connect(iterSolver->inMass());
		this->stateCenter()->connect(iterSolver->inCenter());
		this->stateVelocity()->connect(iterSolver->inVelocity());
		this->stateAngularVelocity()->connect(iterSolver->inAngularVelocity());
		this->stateRotationMatrix()->connect(iterSolver->inRotationMatrix());
		this->stateInertia()->connect(iterSolver->inInertia());
		this->stateQuaternion()->connect(iterSolver->inQuaternion());
		this->stateInitialInertia()->connect(iterSolver->inInitialInertia());

		this->stateBallAndSocketJoints()->connect(iterSolver->inBallAndSocketJoints());
		this->stateSliderJoints()->connect(iterSolver->inSliderJoints());
		this->stateHingeJoints()->connect(iterSolver->inHingeJoints());
		this->stateFixedJoints()->connect(iterSolver->inFixedJoints());

		merge->outContacts()->connect(iterSolver->inContacts());

		this->animationPipeline()->pushModule(iterSolver);
	}

	template<typename TDataType>
	RigidBodySystem<TDataType>::~RigidBodySystem()
	{
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::addBox(
		const BoxInfo& box,
		const RigidBodyInfo& bodyDef, 
		const Real density)
	{
		auto b = box;
		auto bd = bodyDef;

		float lx = 2.0f * b.halfLength[0];
		float ly = 2.0f * b.halfLength[1];
		float lz = 2.0f * b.halfLength[2];
		bd.position = b.center;

		bd.mass = density * lx * ly * lz;
		bd.inertia = 1.0f / 12.0f * bd.mass
			* Mat3f(ly*ly + lz * lz, 0, 0,
				0, lx*lx + lz * lz, 0,
				0, 0, lx*lx + ly * ly);

		bd.shapeType = ET_BOX;
		bd.angle = b.rot;

		mHostRigidBodyStates.insert(mHostRigidBodyStates.begin() + mHostSpheres.size() + mHostBoxes.size(), bd);
		mHostBoxes.push_back(b);
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::addSphere(
		const SphereInfo& sphere, 
		const RigidBodyInfo& bodyDef,
		const Real density /*= Real(1)*/)
	{
		auto b = sphere;
		auto bd = bodyDef;

		bd.position = b.center;

		float r = b.radius;
		if (bd.mass <= 0.0f) {
			bd.mass = 4 / 3.0f*M_PI*r*r*r*density;
		}
		float I11 = r * r;
		bd.inertia = 0.4f * bd.mass
			* Mat3f(I11, 0, 0,
				0, I11, 0,
				0, 0, I11);

		bd.shapeType = ET_SPHERE;
		bd.angle = b.rot;

		mHostRigidBodyStates.insert(mHostRigidBodyStates.begin() + mHostSpheres.size(), bd);
		mHostSpheres.push_back(b);
	}

	template<typename TDataType>
	Mat3f RigidBodySystem<TDataType>::pointInertia(Coord r1)
	{
		Real x = r1.x;
		Real y = r1.y;
		Real z = r1.z;
		return Mat3f(y * y + z * z, -x * y, -x * z, -y * x, x * x + z * z, -y * z, -z * x, -z * y, x * x + y * y);
	}



	template<typename TDataType>
	void RigidBodySystem<TDataType>::addTet(
		const TetInfo& tet,
		const RigidBodyInfo& bodyDef, 
		const Real density /*= Real(1)*/)
	{
		auto b = tet;
		auto bd = bodyDef;

		bd.position = (tet.v[0] + tet.v[1] + tet.v[2] + tet.v[3]) / 4;

		auto centroid = bd.position;

		auto tmpMat = Mat3f(tet.v[1] - tet.v[0], tet.v[2] - tet.v[0], tet.v[3] - tet.v[0]);
		Real volume = (1.0 / 6.0) * abs(tmpMat.determinant());
		Real mass = volume * density;
		bd.mass = mass;


		Mat3f inertiaMatrix(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
		for (int i = 0; i < 4; i++)
		{
			Coord r = tet.v[i] - centroid;
			inertiaMatrix += (mass / 4.0) * pointInertia(r);
		}

		bd.inertia = inertiaMatrix;
		bd.shapeType = ET_TET;
		bd.angle = Quat<Real>();

		mHostRigidBodyStates.insert(mHostRigidBodyStates.begin() + mHostSpheres.size() + mHostBoxes.size() + mHostTets.size(), bd);
		mHostTets.push_back(b);
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::addCapsule(
		const CapsuleInfo& capsule,
		const RigidBodyInfo& bodyDef, 
		const Real density /*= Real(100)*/)
	{
		auto b = capsule;
		auto bd = bodyDef;

		float lx = 2.0f * b.halfLength;
		float ly = 2.0f * b.radius;
		float lz = 2.0f * b.radius;
		bd.position = b.center;

		bd.mass = density * lx * ly * lz;
		bd.inertia = 1.0f / 12.0f * bd.mass
			* Mat3f(ly * ly + lz * lz, 0, 0,
				0, lx * lx + lz * lz, 0,
				0, 0, lx * lx + ly * ly);

		bd.shapeType = ET_CAPSULE;
		bd.angle = b.rot;

		mHostRigidBodyStates.insert(mHostRigidBodyStates.begin() + mHostSpheres.size() + mHostBoxes.size() + mHostTets.size() + mHostCapsules.size(), bd);
		mHostCapsules.push_back(b);
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::addBallAndSocketJoint(const BallAndSocketJoint& joint)
	{
		mHostJointsBallAndSocket.push_back(joint);
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::addSliderJoint(const SliderJoint& joint)
	{
		mHostJointsSlider.push_back(joint);
	}
	template<typename TDataType>
	void RigidBodySystem<TDataType>::addHingeJoint(const HingeJoint& joint)
	{
		mHostJointsHinge.push_back(joint);
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::addFixedJoint(const FixedJoint& joint)
	{
		mHostJointsFixed.push_back(joint);
	}

	template <typename Real, typename Coord, typename Matrix, typename Quat>
	__global__ void RB_SetupInitialStates(
		DArray<Real> mass,
		DArray<Coord> pos,
		DArray<Matrix> rotation,
		DArray<Coord> velocity,
		DArray<Coord> angularVelocity,
		DArray<Quat> rotation_q,
		DArray<Matrix> inertia,
		DArray<CollisionMask> mask,
		DArray<RigidBodyInfo> states,
		ElementOffset offset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= rotation_q.size())
			return;
		
		mass[tId] = states[tId].mass;
		rotation[tId] = states[tId].angle.toMatrix3x3();
		velocity[tId] = states[tId].linearVelocity;
		angularVelocity[tId] = states[tId].angularVelocity;
		rotation_q[tId] = states[tId].angle;
		pos[tId] = states[tId].position;
		inertia[tId] = states[tId].inertia;
		mask[tId] = states[tId].collisionMask;
	}

	__global__ void SetupBoxes(
		DArray<dyno::Box3D> box3d,
		DArray<BoxInfo> boxInfo)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boxInfo.size()) return;

		box3d[tId].center = boxInfo[tId].center;
		box3d[tId].extent = boxInfo[tId].halfLength;

		Mat3f rot = boxInfo[tId].rot.toMatrix3x3();

		box3d[tId].u = rot * Vec3f(1, 0, 0);
		box3d[tId].v = rot * Vec3f(0, 1, 0);
		box3d[tId].w = rot * Vec3f(0, 0, 1);
	}

	__global__ void SetupSpheres(
		DArray<Sphere3D> sphere3d,
		DArray<SphereInfo> sphereInfo)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sphereInfo.size()) return;

		sphere3d[tId].radius = sphereInfo[tId].radius;
		sphere3d[tId].center = sphereInfo[tId].center;
		sphere3d[tId].rotation = sphereInfo[tId].rot;
	}

	__global__ void SetupTets(
		DArray<Tet3D> tet3d,
		DArray<TetInfo> tetInfo)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tetInfo.size()) return;

		tet3d[tId].v[0] = tetInfo[tId].v[0];
		tet3d[tId].v[1] = tetInfo[tId].v[1];
		tet3d[tId].v[2] = tetInfo[tId].v[2];
		tet3d[tId].v[3] = tetInfo[tId].v[3];
	}

	__global__ void SetupCaps(
		DArray<Capsule3D> cap3d,
		DArray<CapsuleInfo> capInfo)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= capInfo.size()) return;

		float halfLenght = capInfo[tId].halfLength;
		Mat3f rot = capInfo[tId].rot.toMatrix3x3();

		cap3d[tId].segment.v0 = capInfo[tId].center - rot * Vec3f(0, 0, halfLenght);
		cap3d[tId].segment.v1 = capInfo[tId].center + rot * Vec3f(0, 0, halfLenght);
		cap3d[tId].radius = capInfo[tId].radius;
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::resetStates()
	{
		auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->stateTopology()->getDataPtr());

		mDeviceBoxes.assign(mHostBoxes);
		mDeviceSpheres.assign(mHostSpheres);
		mDeviceTets.assign(mHostTets);
		mDeviceCapsules.assign(mHostCapsules);

		this->stateBallAndSocketJoints()->assign(mHostJointsBallAndSocket);
		this->stateSliderJoints()->assign(mHostJointsSlider);
		this->stateHingeJoints()->assign(mHostJointsHinge);
		this->stateFixedJoints()->assign(mHostJointsFixed);

		auto& boxes = topo->getBoxes();
		auto& spheres = topo->getSpheres();
		auto& tets = topo->getTets();
		auto& caps = topo->getCaps();

		boxes.resize(mDeviceBoxes.size());
		spheres.resize(mDeviceSpheres.size());
		tets.resize(mDeviceTets.size());
		caps.resize(mDeviceCapsules.size());

		//Setup the topology
		cuExecute(mDeviceBoxes.size(),
			SetupBoxes,
			boxes,
			mDeviceBoxes);

		cuExecute(mDeviceSpheres.size(),
			SetupSpheres,
			spheres,
			mDeviceSpheres);

		cuExecute(mDeviceTets.size(),
			SetupTets,
			tets,
			mDeviceTets);

		cuExecute(mDeviceCapsules.size(),
			SetupCaps,
			caps,
			mDeviceCapsules);

		mDeviceRigidBodyStates.assign(mHostRigidBodyStates);

		int sizeOfRigids = topo->totalSize();

		ElementOffset eleOffset = topo->calculateElementOffset();

		this->stateRotationMatrix()->resize(sizeOfRigids);
		this->stateAngularVelocity()->resize(sizeOfRigids);
		this->stateCenter()->resize(sizeOfRigids);
		this->stateVelocity()->resize(sizeOfRigids);
		this->stateMass()->resize(sizeOfRigids);
		this->stateInertia()->resize(sizeOfRigids);
		this->stateQuaternion()->resize(sizeOfRigids);
		this->stateCollisionMask()->resize(sizeOfRigids);

		cuExecute(sizeOfRigids,
			RB_SetupInitialStates,
			this->stateMass()->getData(),
			this->stateCenter()->getData(),
			this->stateRotationMatrix()->getData(),
			this->stateVelocity()->getData(),
			this->stateAngularVelocity()->getData(),
			this->stateQuaternion()->getData(),
			this->stateInertia()->getData(),
			this->stateCollisionMask()->getData(),
			mDeviceRigidBodyStates,
			eleOffset);

		this->stateInitialInertia()->resize(sizeOfRigids);
		this->stateInitialInertia()->getDataPtr()->assign(this->stateInertia()->getData());

		updateTopology();

		m_yaw = 0.0f;
		m_pitch = 0.0f;
		m_roll = 0.0f;
		m_recoverSpeed = 0.3f;
	}
	
	template <typename Real, typename Coord>
	__global__ void UpdateSpheres(
		DArray<Sphere3D> sphere,
		DArray<SphereInfo> sphere_init,
		DArray<Coord> pos,
		DArray<Quat<Real>> quat,
		int start_sphere)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= sphere.size()) return;

		sphere[pId].center = pos[pId + start_sphere];
		sphere[pId].rotation = quat[pId + start_sphere];
	}

	template <typename Coord, typename Matrix>
	__global__ void UpdateBoxes(
		DArray<Box3D> box,
		DArray<BoxInfo> box_init,
		DArray<Coord> pos,
		DArray<Matrix> rotation,
		int start_box)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= box.size()) return;
		box[pId].center = pos[pId + start_box];

		box[pId].extent = box_init[pId].halfLength;

		box[pId].u = rotation[pId + start_box] * Coord(1, 0, 0);
		box[pId].v = rotation[pId + start_box] * Coord(0, 1, 0);
		box[pId].w = rotation[pId + start_box] * Coord(0, 0, 1);
	}

	template <typename Coord, typename Matrix>
	__global__ void UpdateTets(
		DArray<Tet3D> tet,
		DArray<TetInfo> tet_init,
		DArray<Coord> pos,
		DArray<Matrix> rotation,
		int start_tet)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= tet.size()) return;

		Coord center_init = (tet_init[pId].v[0] + tet_init[pId].v[1] + tet_init[pId].v[2] + tet_init[pId].v[3]) / 4.0f;
		tet[pId].v[0] = rotation[pId + start_tet] * (tet_init[pId].v[0] - center_init) + pos[pId + start_tet];
		tet[pId].v[1] = rotation[pId + start_tet] * (tet_init[pId].v[1] - center_init) + pos[pId + start_tet];
		tet[pId].v[2] = rotation[pId + start_tet] * (tet_init[pId].v[2] - center_init) + pos[pId + start_tet];
		tet[pId].v[3] = rotation[pId + start_tet] * (tet_init[pId].v[3] - center_init) + pos[pId + start_tet];
	}

	template <typename Real, typename Coord>
	__global__ void UpdateCapsules(
		DArray<Capsule3D> caps,
		DArray<CapsuleInfo> cap_init,
		DArray<Coord> pos,
		DArray<Quat<Real>> quat,
		int start_cap)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= caps.size()) return;

		Capsule3D cap;
		cap.radius = cap_init[pId].radius;
		cap.segment.v0 = pos[pId + start_cap] - quat[pId + start_cap].rotate(Coord(0, 0, cap_init[pId].halfLength));
		cap.segment.v1 = pos[pId + start_cap] + quat[pId + start_cap].rotate(Coord(0, 0, cap_init[pId].halfLength));

		caps[pId] = cap;
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::updateTopology()
	{
		auto discreteSet = TypeInfo::cast<DiscreteElements<DataType3f>>(this->stateTopology()->getDataPtr());

		ElementOffset offset = discreteSet->calculateElementOffset();

		CArray<Coord> hPos;
		hPos.assign(this->stateCenter()->constData());

		cuExecute(mDeviceBoxes.size(),
			UpdateBoxes,
			discreteSet->getBoxes(),
			mDeviceBoxes,
			this->stateCenter()->getData(),
			this->stateRotationMatrix()->getData(),
			offset.boxIndex());

		cuExecute(mDeviceSpheres.size(),
			UpdateSpheres,
			discreteSet->getSpheres(),
			mDeviceSpheres,
			this->stateCenter()->getData(),
			this->stateQuaternion()->getData(),
			offset.sphereIndex());

		cuExecute(mDeviceTets.size(),
			UpdateTets,
			discreteSet->getTets(),
			mDeviceTets,
			this->stateCenter()->getData(),
			this->stateRotationMatrix()->getData(),
			offset.tetIndex());

		cuExecute(mDeviceCapsules.size(),
			UpdateCapsules,
			discreteSet->getCaps(),
			mDeviceCapsules,
			this->stateCenter()->getData(),
			this->stateQuaternion()->getData(),
			offset.capsuleIndex());
	}

	DEFINE_CLASS(RigidBodySystem);
}