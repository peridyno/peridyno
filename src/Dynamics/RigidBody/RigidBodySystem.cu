#include "RigidBodySystem.h"

#include "Topology/Primitive3D.h"
#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"
#include <cuda_runtime.h>
#include "IterativeConstraintSolver.h"

//Module headers
#include "ContactsUnion.h"


namespace dyno
{
	IMPLEMENT_TCLASS(RigidBodySystem, TDataType)

	typedef typename TOrientedBox3D<Real> Box3D;

	template<typename TDataType>
	RigidBodySystem<TDataType>::RigidBodySystem(std::string name)
		: Node(name)
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
		this->varTimeStep()->connect(iterSolver->inTimeStep());
		this->varFrictionEnabled()->connect(iterSolver->varFrictionEnabled());
		this->stateMass()->connect(iterSolver->inMass());
		this->stateCenter()->connect(iterSolver->inCenter());
		this->stateVelocity()->connect(iterSolver->inVelocity());
		this->stateAngularVelocity()->connect(iterSolver->inAngularVelocity());
		this->stateRotationMatrix()->connect(iterSolver->inRotationMatrix());
		this->stateInertia()->connect(iterSolver->inInertia());
		this->stateQuaternion()->connect(iterSolver->inQuaternion());
		this->stateInitialInertia()->connect(iterSolver->inInitialInertia());

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
			bd.mass = 3 / 4.0f*M_PI*r*r*r*density;
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
	void RigidBodySystem<TDataType>::addTet(
		const TetInfo& tet,
		const RigidBodyInfo& bodyDef, 
		const Real density /*= Real(1)*/)
	{
		auto b = tet;
		auto bd = bodyDef;

		bd.position = (tet.v[0] + tet.v[1] + tet.v[2] + tet.v[3]) / 4;

		float r = 0.025;
		if (bd.mass <= 0.0f) {
			bd.mass = 3 / 4.0f*M_PI*r*r*r*density;
		}
		float I11 = r * r;
		bd.inertia = 0.4f * bd.mass
			* Mat3f(I11, 0, 0,
				0, I11, 0,
				0, 0, I11);

		bd.shapeType = ET_TET;
		bd.angle = Quat<Real>();

		mHostRigidBodyStates.insert(mHostRigidBodyStates.begin() + mHostSpheres.size() + mHostBoxes.size() + mHostTets.size(), bd);
		mHostTets.push_back(b);
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
		DArray<Box3D> box3d,
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

	template<typename TDataType>
	void RigidBodySystem<TDataType>::resetStates()
	{
		auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->stateTopology()->getDataPtr());

		mDeviceBoxes.assign(mHostBoxes);
		mDeviceSpheres.assign(mHostSpheres);
		mDeviceTets.assign(mHostTets);

		auto& boxes = topo->getBoxes();
		auto& spheres = topo->getSpheres();
		auto& tets = topo->getTets();

		boxes.resize(mDeviceBoxes.size());
		spheres.resize(mDeviceSpheres.size());
		tets.resize(mDeviceTets.size());

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

		mDeviceRigidBodyStates.assign(mHostRigidBodyStates);

		int sizeOfRigids = topo->totalSize();

		ElementOffset eleOffset = topo->calculateElementOffset();

		this->stateRotationMatrix()->setElementCount(sizeOfRigids);
		this->stateAngularVelocity()->setElementCount(sizeOfRigids);
		this->stateCenter()->setElementCount(sizeOfRigids);
		this->stateVelocity()->setElementCount(sizeOfRigids);
		this->stateMass()->setElementCount(sizeOfRigids);
		this->stateInertia()->setElementCount(sizeOfRigids);
		this->stateQuaternion()->setElementCount(sizeOfRigids);
		this->stateCollisionMask()->setElementCount(sizeOfRigids);

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

		this->stateInitialInertia()->setElementCount(sizeOfRigids);
		this->stateInitialInertia()->getDataPtr()->assign(this->stateInertia()->getData());



		m_yaw = 0.0f;
		m_pitch = 0.0f;
		m_roll = 0.0f;
		m_recoverSpeed = 0.3f;
	}
	
	template <typename Coord>
	__global__ void UpdateSpheres(
		DArray<Sphere3D> sphere,
		DArray<Coord> pos,
		int start_sphere)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= sphere.size()) return;

		sphere[pId].center = pos[pId + start_sphere];
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

		Coord3D center_init = (tet_init[pId].v[0] + tet_init[pId].v[1] + tet_init[pId].v[2] + tet_init[pId].v[3]) / 4.0f;
		tet[pId].v[0] = rotation[pId + start_tet] * (tet_init[pId].v[0] - center_init) + pos[pId + start_tet];
		tet[pId].v[1] = rotation[pId + start_tet] * (tet_init[pId].v[1] - center_init) + pos[pId + start_tet];
		tet[pId].v[2] = rotation[pId + start_tet] * (tet_init[pId].v[2] - center_init) + pos[pId + start_tet];
		tet[pId].v[3] = rotation[pId + start_tet] * (tet_init[pId].v[3] - center_init) + pos[pId + start_tet];
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::updateTopology()
	{
		auto discreteSet = TypeInfo::cast<DiscreteElements<DataType3f>>(this->stateTopology()->getDataPtr());

		ElementOffset offset = discreteSet->calculateElementOffset();

		cuExecute(mDeviceBoxes.size(),
			UpdateBoxes,
			discreteSet->getBoxes(),
			mDeviceBoxes,
			this->stateCenter()->getData(),
			this->stateRotationMatrix()->getData(),
			offset.boxIndex());

		cuExecute(mDeviceBoxes.size(),
			UpdateSpheres,
			discreteSet->getSpheres(),
			this->stateCenter()->getData(),
			offset.sphereIndex());

		cuExecute(mDeviceTets.size(),
			UpdateTets,
			discreteSet->getTets(),
			mDeviceTets,
			this->stateCenter()->getData(),
			this->stateRotationMatrix()->getData(),
			offset.tetIndex());
	}
	//myCode---------------------------------
	
	template<typename TQuat, typename Matrix>
	__global__ void updateVelocityAngulessss(
		DArray<Vec3f> m_velocity,
		DArray<Vec3f> m_angularvelocity,
		DArray<TQuat> quat,
		DArray<float> m_mass,
		DArray<Matrix> m_inertia,
		Vec3f force,
		Vec3f torque,
		float dt
		)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= m_velocity.size()) return;
		
		Quat<float> quat1 = Quat<float>(quat[tId].w, quat[tId].x, quat[tId].z, quat[tId].y);

		Matrix m_rot = quat1.toMatrix3x3();

		m_velocity[0] += dt * force / m_mass[0];// +dt * m_acceleration * m_rot * Vec3f(0.0f, 0.0f, -1.0f);
		m_angularvelocity[0] +=dt * m_inertia[0].inverse() * m_rot.transpose() * torque;

		Vec3f local_v = m_rot.transpose() * m_velocity[0];
		local_v.x *= 0.5f;
		float m_damping = 0.9f;
		local_v.z *= m_damping;

		m_velocity[0] = m_rot * local_v;
		m_angularvelocity[0] *= m_damping;
		//printf("come come le \n");
	}
	template<typename TDataType>
	void RigidBodySystem<TDataType>::updateVelocityAngule(Vec3f force, Vec3f torque, float dt)
	{
		DArray<Vec3f> mm_velocity = stateVelocity()->getData();
		DArray<Vec3f> mm_AngularVelocity = stateAngularVelocity()->getData();
		DArray<TQuat> mm_rot = stateQuaternion()->getData();
		DArray<float> mm_mass = stateMass()->getData();
		DArray<Matrix> mm_inertia = stateInertia()->getData();
		
		//DEF_ARRAY_STATE(Matrix, Inertia
		cuExecute(mm_velocity.size(),
			updateVelocityAngulessss,
			mm_velocity,
			mm_AngularVelocity,
			mm_rot,
			mm_mass,
			mm_inertia,
			force, 
			torque, 
			dt
			);
	}

	template<typename TQuat>
	__global__ void m_advect(
		DArray<Vec3f> m_velocity,
		DArray<Vec3f> m_center,
		DArray<Vec3f> m_AngularVelocity,
		DArray<TQuat> m_quaternion,
		float dt
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= m_velocity.size()) return;

		m_center[tId] += dt * (m_velocity[tId]);
		m_quaternion[tId] = m_quaternion[tId] + (0.5f * dt) * Quat<float>(0, m_AngularVelocity[tId].x, m_AngularVelocity[tId].y, m_AngularVelocity[tId].z) * m_quaternion[tId];

		m_quaternion[tId] = m_quaternion[tId] / m_quaternion[tId].norm();

	}
	template<typename TDataType>
	void RigidBodySystem<TDataType>::getEulerAngle(float& yaw, float& pitch, float& roll)
	{
		DArray<TQuat> mm_quaternion = stateQuaternion()->getData();
		CArray<TQuat> c_quaternion;
		c_quaternion.resize(mm_quaternion.size());
		c_quaternion.assign(mm_quaternion);

		TQuat m_quaternion = c_quaternion[0];
		//该系统实现y轴朝上，标准yaw, pitch, roll则是z轴朝下，因为计算之前需要先绕着x轴旋转90度。
		Quat<float> quat = m_quaternion * Quat<float>(cos(M_PI / 4.0f), sin(M_PI / 4.0f), 0.0f, 0.0f);

		double sinr_cosp = +2.0 * (quat.w * quat.x + quat.y * quat.z);
		double cosr_cosp = +1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y);
		//减去90度
		roll = atan2(sinr_cosp, cosr_cosp) - M_PI / 2.0f;

		// pitch (y-axis rotation)
		double sinp = +2.0 * (quat.w * quat.y - quat.z * quat.x);
		if (fabs(sinp) >= 1)
			pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
		else
			pitch = asin(sinp);

		// yaw (z-axis rotation)
		double siny_cosp = +2.0 * (quat.w * quat.z + quat.x * quat.y);
		double cosy_cosp = +1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z);
		yaw = atan2(siny_cosp, cosy_cosp);
	}

	template<typename TQuat>
	__global__ void m_getQuaternian(
		DArray<TQuat> m_quaternion,
		float yaw,
		float pitch,
		float roll
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= m_quaternion.size()) return;
		/*
		double cy = cos(yaw * 0.5);
		double sy = sin(yaw * 0.5);
		double cp = cos(pitch * 0.5);
		double sp = sin(pitch * 0.5);
		double cr = cos(roll * 0.5);
		double sr = sin(roll * 0.5);

		m_quaternion[0].w = cy * cp * cr + sy * sp * sr;
		m_quaternion[0].x = cy * cp * sr - sy * sp * cr;
		m_quaternion[0].y = sy * cp * sr + cy * sp * cr;
		m_quaternion[0].z = sy * cp * cr - cy * sp * sr;
		*/
		m_quaternion[tId] = Quat<float>(yaw, pitch, roll);

	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::advect(float dt) {
		DArray<Vec3f> mm_velocity = stateVelocity()->getData();
		DArray<Vec3f> mm_center = stateCenter()->getData();
		DArray<Vec3f> mm_AngularVelocity = stateAngularVelocity()->getData();
		DArray<TQuat> mm_quaternion = stateQuaternion()->getData();
		cuExecute(mm_velocity.size(),
			m_advect,
			mm_velocity,
			mm_center,
			mm_AngularVelocity,
			mm_quaternion,
			dt
		);

		getEulerAngle(m_yaw, m_pitch, m_roll);
		m_roll *= (1.0f - m_recoverSpeed);
		m_pitch *= (1.0f - m_recoverSpeed);

		cuExecute(mm_quaternion.size(),
			m_getQuaternian,
			mm_quaternion,
			m_yaw,
			m_pitch,
			m_roll
		);
	}
	
	DEFINE_CLASS(RigidBodySystem);
}