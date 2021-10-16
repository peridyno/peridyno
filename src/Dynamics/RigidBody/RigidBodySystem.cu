#include "RigidBodySystem.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(RigidBodySystem, TDataType)

	typedef typename TOrientedBox3D<Real> Box3D;

	template<typename TDataType>
	RigidBodySystem<TDataType>::RigidBodySystem(std::string name)
		: Node(name)
	{
		auto defaultTopo = std::make_shared<DiscreteElements<TDataType>>();
		this->currentTopology()->setDataPtr(std::make_shared<DiscreteElements<TDataType>>());

		mElementQuery = std::make_shared<NeighborElementQuery<TDataType>>();
		this->currentTopology()->connect(mElementQuery->inDiscreteElements());
		this->stateCollisionMask()->connect(mElementQuery->inCollisionMask());
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
		auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->currentTopology()->getDataPtr());

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

		this->currentRotationMatrix()->setElementCount(sizeOfRigids);
		this->currentAngularVelocity()->setElementCount(sizeOfRigids);
		this->currentCenter()->setElementCount(sizeOfRigids);
		this->currentVelocity()->setElementCount(sizeOfRigids);
		this->currentMass()->setElementCount(sizeOfRigids);
		this->currentInertia()->setElementCount(sizeOfRigids);
		this->currentQuaternion()->setElementCount(sizeOfRigids);
		this->stateCollisionMask()->setElementCount(sizeOfRigids);

		nbrContacts.resize(sizeOfRigids);

		mBoundaryContactCounter.resize(sizeOfRigids);

		cuExecute(sizeOfRigids,
			RB_SetupInitialStates,
			this->currentMass()->getData(),
			this->currentCenter()->getData(),
			this->currentRotationMatrix()->getData(),
			this->currentVelocity()->getData(),
			this->currentAngularVelocity()->getData(),
			this->currentQuaternion()->getData(),
			this->currentInertia()->getData(),
			this->stateCollisionMask()->getData(),
			mDeviceRigidBodyStates,
			eleOffset);

		mInitialInertia.assign(this->currentInertia()->getData());
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

	template <typename Coord, typename Matrix, typename Quat>
	__global__ void RB_UpdateGesture(
		DArray<Coord> pos,
		DArray<Quat> rotQuat,
		DArray<Matrix> rotMat,
		DArray<Matrix> inertia,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Matrix> inertia_init,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;
		
		pos[pId] += velocity[pId] * dt;

		rotQuat[pId] = rotQuat[pId].normalize();
		rotMat[pId] = rotQuat[pId].toMatrix3x3();
		
		rotQuat[pId] += dt * 0.5f * 
			Quat(angular_velocity[pId][0], angular_velocity[pId][1],angular_velocity[pId][2], 0.0)
			*(rotQuat[pId]);

		inertia[pId] = rotMat[pId] * inertia_init[pId] * rotMat[pId].inverse();
	}

	template <typename Coord>
	__global__ void RB_UpdateVelocity(
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> accel,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= accel.size() / 2) return;

		velocity[pId] += accel[2 * pId] * dt;
		velocity[pId] += Coord(0, -9.8f, 0) * dt;

		angular_velocity[pId] += accel[2 * pId + 1] * dt;
	}

	template <typename Coord, typename Matrix, typename ContactPair>
	__global__ void CalculateJacobians(
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Coord> pos,
		DArray<Matrix> inertia,
		DArray<Real> mass,
		DArray<ContactPair> nbc)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbc[pId].bodyId1;
		int idx2 = nbc[pId].bodyId2;

		//printf("%d %d\n", idx1, idx2);

		if (nbc[pId].contactType == ContactType::CT_NONPENETRATION) // contact, collision
		{
			Coord p1 = nbc[pId].pos1;
			Coord p2 = nbc[pId].pos2;
			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];
			Coord r2 = p2 - pos[idx2];

			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			J[4 * pId + 2] = -n;
			J[4 * pId + 3] = -(r2.cross(n));

			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			B[4 * pId + 2] = -n / mass[idx2];
			B[4 * pId + 3] = inertia[idx2].inverse() * (-r2.cross(n));
		}
		else if (nbc[pId].contactType == ContactType::CT_BOUDNARY) // boundary
		{
			Coord p1 = nbc[pId].pos1;
		//	printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ %d %.3lf %.3lf %.3lf\n", idx1, p1[0], p1[1], p1[2]);

			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];
			

			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			J[4 * pId + 2] = Coord(0);
			J[4 * pId + 3] = Coord(0);

			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			B[4 * pId + 2] = Coord(0);
			B[4 * pId + 3] = Coord(0);
		}
		else if (nbc[pId].contactType == ContactType::CT_FRICTION) // friction
		{
			Coord p1 = nbc[pId].pos1;
			//printf("~~~~~~~ %.3lf %.3lf %.3lf\n", p1[0], p1[1], p1[2]);

			
			Coord p2 = Coord(0);
			if(idx2 != -1)
				p2 = nbc[pId].pos2;

			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];
			Coord r2 = Coord(0);
			if (idx2 != -1)
				r2 = p2 - pos[idx2];

			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			if(idx2 != -1)
			{ 
				J[4 * pId + 2] = -n;
				J[4 * pId + 3] = -(r2.cross(n));
			}
			else
			{
				J[4 * pId + 2] = Coord(0);
				J[4 * pId + 3] = Coord(0);
			}
			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			if(idx2 != -1)
			{ 
				B[4 * pId + 2] = -n / mass[idx2];
				B[4 * pId + 3] = inertia[idx2].inverse() * (-r2.cross(n));
			}
			else
			{
				B[4 * pId + 2] = Coord(0);
				B[4 * pId + 3] = Coord(0);
			}
		}
	}


	template <typename ContactPair>
	__global__ void CalculateNbrCons(
		DArray<ContactPair> nbc,
		DArray<Real> nbrCnt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbc.size()) return;

		int idx1 = nbc[pId].bodyId1;
		int idx2 = nbc[pId].bodyId2;

		if(idx1 != -1)
			atomicAdd(&nbrCnt[idx1], 1.0f);
		if(idx2 != -1)
			atomicAdd(&nbrCnt[idx2], 1.0f);
	}

	template <typename Coord, typename Matrix, typename ContactPair>
	__global__ void CalculateJacobians(
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Coord> pos,
		DArray<Matrix> inertia,
		DArray<Matrix> inertia_eq,
		DArray<Real> mass,
		DArray<Real> mass_eq,
		DArray<ContactPair> nbc)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbc[pId].bodyId1;
		int idx2 = nbc[pId].bodyId2;

		//printf("%d %d\n", idx1, idx2);
		//EPSILON

		if (nbc[pId].contactType == ContactType::CT_NONPENETRATION) // contact, collision
		{
			Coord p1 = nbc[pId].pos1;
			Coord p2 = nbc[pId].pos2;
			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];
			Coord r2 = p2 - pos[idx2];

			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			J[4 * pId + 2] = -n;
			J[4 * pId + 3] = -(r2.cross(n));

			n /= n.norm();
			Coord3D n1(0), n2(0);

			Real ratio1 = 1.0f;
			Real ratio2 = 1.0f;

			if (n[0] > EPSILON)
			{
				n1 += Coord(n[0], 0, 0) / mass_eq[6 * idx1 + 1];
				//ratio1 += mass[idx1] / mass_eq[6 * idx1 + 1];
				ratio1 = min(ratio1, mass[idx1] / mass_eq[6 * idx1 + 1]);
				n2 -= Coord(n[0], 0, 0) / mass_eq[6 * idx2 + 0];
				//ratio2 += mass[idx2] / mass_eq[6 * idx2 + 0];
				ratio2 = min(ratio2, mass[idx2] / mass_eq[6 * idx2 + 0]);

			}
			else
			{
				n1 += Coord(n[0], 0, 0) / mass_eq[6 * idx1 + 0];
				//ratio1 += mass[idx1] / mass_eq[6 * idx1 + 0];
				ratio1 = min(ratio1, mass[idx1] / mass_eq[6 * idx1 + 0]);
				n2 -= Coord(n[0], 0, 0) / mass_eq[6 * idx2 + 1];
				//ratio2 += mass[idx2] / mass_eq[6 * idx2 + 1];
				ratio2 = min(ratio2, mass[idx2] / mass_eq[6 * idx2 + 1]);
			}
			if (n[1] > EPSILON)
			{
				n1 += Coord(0, n[1], 0) / mass_eq[6 * idx1 + 3];
				//ratio1 += mass[idx1] / mass_eq[6 * idx1 + 3];
				ratio1 = min(ratio1, mass[idx1] / mass_eq[6 * idx1 + 3]);
				n2 -= Coord(0, n[1], 0) / mass_eq[6 * idx2 + 2];
				//ratio2 += mass[idx2] / mass_eq[6 * idx2 + 2];
				ratio2 = min(ratio2, mass[idx2] / mass_eq[6 * idx2 + 2]);
			}
			else
			{
				n1 += Coord(0, n[1], 0) / mass_eq[6 * idx1 + 2];
				//ratio1 += mass[idx1] / mass_eq[6 * idx1 + 2];
				ratio1 = min(ratio1, mass[idx1] / mass_eq[6 * idx1 + 2]);
				n2 -= Coord(0, n[1], 0) / mass_eq[6 * idx2 + 3];
				//ratio2 += mass[idx2] / mass_eq[6 * idx2 + 3];
				ratio2 = min(ratio2, mass[idx2] / mass_eq[6 * idx2 + 3]);
			}
			if (n[2] > EPSILON)
			{
				n1 += Coord(0, 0, n[2]) / mass_eq[6 * idx1 + 5];
				//ratio1 += mass[idx1] / mass_eq[6 * idx1 + 5];
				ratio1 = min(ratio1, mass[idx1] / mass_eq[6 * idx1 + 5]);
				n2 -= Coord(0, 0, n[2]) / mass_eq[6 * idx2 + 4];
				//ratio2 += mass[idx2] / mass_eq[6 * idx2 + 4];
				ratio2 = min(ratio2, mass[idx2] / mass_eq[6 * idx2 + 4]);
			}
			else
			{
				n1 += Coord(0, 0, n[2]) / mass_eq[6 * idx1 + 4];
				//ratio1 += mass[idx1] / mass_eq[6 * idx1 + 4];
				ratio1 = min(ratio1, mass[idx1] / mass_eq[6 * idx1 + 4]);
				n2 -= Coord(0, 0, n[2]) / mass_eq[6 * idx2 + 5];
				//ratio2 += mass[idx2] / mass_eq[6 * idx2 + 5];
				ratio2 = min(ratio2, mass[idx2] / mass_eq[6 * idx2 + 5]);
			}

			ratio1 /= 3.0f;
			ratio2 /= 3.0f;

			printf("%.5lf %.5lf %.5lf %.5lf\n", ratio1, ratio2, mass[idx1], mass[idx2]);

			B[4 * pId] = n1;//n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n)) * ratio1;
			B[4 * pId + 2] = n2;//-n / mass[idx2];
			B[4 * pId + 3] = inertia[idx2].inverse() * (-r2.cross(n)) * ratio2;
		}

		else if (nbc[pId].contactType == ContactType::CT_BOUDNARY) // boundary
		{

			Coord p1 = nbc[pId].pos1;
			//	printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ %d %.3lf %.3lf %.3lf\n", idx1, p1[0], p1[1], p1[2]);

			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];


			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			J[4 * pId + 2] = Coord(0);
			J[4 * pId + 3] = Coord(0);

			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			B[4 * pId + 2] = Coord(0);
			B[4 * pId + 3] = Coord(0);
		}

		else if (nbc[pId].contactType == ContactType::CT_FRICTION) // friction
		{
			Coord p1 = nbc[pId].pos1;
			//printf("~~~~~~~ %.3lf %.3lf %.3lf\n", p1[0], p1[1], p1[2]);


			Coord p2 = Coord(0);
			if (idx2 != -1)
				p2 = nbc[pId].pos2;

			Coord n = nbc[pId].normal1;
			Coord r1 = p1 - pos[idx1];
			Coord r2 = Coord(0);
			if (idx2 != -1)
				r2 = p2 - pos[idx2];

			J[4 * pId] = n;
			J[4 * pId + 1] = (r1.cross(n));
			if (idx2 != -1)
			{
				J[4 * pId + 2] = -n;
				J[4 * pId + 3] = -(r2.cross(n));
			}
			else
			{
				J[4 * pId + 2] = Coord(0);
				J[4 * pId + 3] = Coord(0);
			}
			B[4 * pId] = n / mass[idx1];
			B[4 * pId + 1] = inertia[idx1].inverse() * (r1.cross(n));
			if (idx2 != -1)
			{
				B[4 * pId + 2] = -n / mass[idx2];
				B[4 * pId + 3] = inertia[idx2].inverse() * (-r2.cross(n));
			}
			else
			{
				B[4 * pId + 2] = Coord(0);
				B[4 * pId + 3] = Coord(0);
			}
		}

	}



	// ignore zeta !!!!!!
	template <typename Coord, typename ContactPair>
	__global__ void CalculateEta(
		DArray<Real> eta,
		DArray<Coord> velocity,
		DArray<Coord> angular_velocity,
		DArray<Coord> J,
		DArray<Real> mass,
		DArray<ContactPair> nbq,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbq[pId].bodyId1;
		int idx2 = nbq[pId].bodyId2;
		//printf("from ita %d\n", pId);
		Real ita_i = Real(0);
		if (true) // test dist constraint
		{
			ita_i -= J[4 * pId].dot(velocity[idx1]);
			ita_i -= J[4 * pId + 1].dot(angular_velocity[idx1]);
			if(idx2 != -1)
			{ 
				ita_i -= J[4 * pId + 2].dot(velocity[idx2]);
				ita_i -= J[4 * pId + 3].dot(angular_velocity[idx2]);
			}
		}
		eta[pId] = ita_i / dt;
		if (nbq[pId].contactType == ContactType::CT_NONPENETRATION || nbq[pId].contactType == ContactType::CT_BOUDNARY)
		{
			eta[pId] += min(nbq[pId].interpenetration, nbq[pId].interpenetration) / dt / dt / 15.0f;
		}

	}

	template <typename Coord>
	__global__ void CalculateDiagonals(
		DArray<Real> D,
		DArray<Coord> J,
		DArray<Coord> B)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= J.size() / 4) return;

		Real d = Real(0);
		d += J[4 * tId].dot(B[4 * tId]);
		d += J[4 * tId + 1].dot(B[4 * tId + 1]);
		d += J[4 * tId + 2].dot(B[4 * tId + 2]);
		d += J[4 * tId + 3].dot(B[4 * tId + 3]);

		D[tId] = d;
	}

	template <typename Coord, typename ContactPair>
	__global__ void TakeOneJacobiIteration(
		DArray<Real> lambda,
		DArray<Coord> accel,
		DArray<Real> d,
		DArray<Coord> J,
		DArray<Coord> B,
		DArray<Real> eta,
		DArray<Real> mass,
		DArray<ContactPair> nbq,
		DArray<Real> stepInv)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= J.size() / 4) return;

		int idx1 = nbq[pId].bodyId1;
		int idx2 = nbq[pId].bodyId2;

		Real eta_i = eta[pId];
		{
			eta_i -= J[4 * pId].dot(accel[idx1 * 2]);
			eta_i -= J[4 * pId + 1].dot(accel[idx1 * 2 + 1]);
			if (idx2 != -1)
			{
				eta_i -= J[4 * pId + 2].dot(accel[idx2 * 2]);
				eta_i -= J[4 * pId + 3].dot(accel[idx2 * 2 + 1]);
			}
		}

		if (d[pId] > EPSILON)
		{
			Real delta_lambda = eta_i / d[pId];
			Real stepInverse = stepInv[idx1];
			if(idx2 != -1)
				stepInverse += stepInv[idx2];
			delta_lambda *= (1.0f / stepInverse);

			//printf("delta_lambda = %.3lf\n", delta_lambda);

			if (nbq[pId].contactType == ContactType::CT_NONPENETRATION || nbq[pId].contactType == ContactType::CT_BOUDNARY) //	PROJECTION!!!!
			{
				Real lambda_new = lambda[pId] + delta_lambda;
				if (lambda_new < 0) lambda_new = 0;

				Real mass_i = mass[idx1];
				if (idx2 != -1)
					mass_i += mass[idx2];

				if (lambda_new > 25 * (mass_i / 0.1)) lambda_new = 25 * (mass_i / 0.1);
				delta_lambda = lambda_new - lambda[pId];
			}

			if (nbq[pId].contactType == ContactType::CT_FRICTION) //	PROJECTION!!!!
			{
				Real lambda_new = lambda[pId] + delta_lambda;
				Real mass_i = mass[idx1];
				if (idx2 != -1)
					mass_i += mass[idx2];

				//if ((lambda_new) > 5 * (mass_i)) lambda_new = 5 * (mass_i);
				//if ((lambda_new) < -5 * (mass_i)) lambda_new = -5 * (mass_i);
				delta_lambda = lambda_new - lambda[pId];
			}

			lambda[pId] += delta_lambda;

			//printf("inside iteration: %d %d %.5lf   %.5lf\n", idx1, idx2, nbq[pId].s4, delta_lambda);

			atomicAdd(&accel[idx1 * 2][0], B[4 * pId][0] * delta_lambda);
			atomicAdd(&accel[idx1 * 2][1], B[4 * pId][1] * delta_lambda);
			atomicAdd(&accel[idx1 * 2][2], B[4 * pId][2] * delta_lambda);

			atomicAdd(&accel[idx1 * 2 + 1][0], B[4 * pId + 1][0] * delta_lambda);
			atomicAdd(&accel[idx1 * 2 + 1][1], B[4 * pId + 1][1] * delta_lambda);
			atomicAdd(&accel[idx1 * 2 + 1][2], B[4 * pId + 1][2] * delta_lambda);

			if (idx2 != -1)
			{
				atomicAdd(&accel[idx2 * 2][0], B[4 * pId + 2][0] * delta_lambda);
				atomicAdd(&accel[idx2 * 2][1], B[4 * pId + 2][1] * delta_lambda);
				atomicAdd(&accel[idx2 * 2][2], B[4 * pId + 2][2] * delta_lambda);

				atomicAdd(&accel[idx2 * 2 + 1][0], B[4 * pId + 3][0] * delta_lambda);
				atomicAdd(&accel[idx2 * 2 + 1][1], B[4 * pId + 3][1] * delta_lambda);
				atomicAdd(&accel[idx2 * 2 + 1][2], B[4 * pId + 3][2] * delta_lambda);
			}
		}

	}

	template<typename ContactPair>
	__global__ void RB_update_offset(
		DArray<ContactPair> nbq,
		int offset)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= nbq.size()) return;
		if(nbq[pId].bodyId1 != -1)
			nbq[pId].bodyId1 += offset;
		if(nbq[pId].bodyId2 != -1)
			nbq[pId].bodyId2 += offset;
	}

	template <typename Coord, typename ContactPair>
	__global__ void SetupContactsWithBoundary(
		DArray<Sphere3D> sphere,
		DArray<Box3D> box,
		DArray<Tet3D> tet,
		DArray<int> count,
		DArray<ContactPair> nbq,
		Coord hi,
		Coord lo,
		int start_sphere,
		int start_box,
		int start_tet,
		int start_segment)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= sphere.size() + box.size() + tet.size()) return;
		
		if (pId < start_box && pId >= start_sphere)//sphere
		{
			int cnt = 0;
			int start_i = count[pId];

			Sphere3D sp = sphere[pId - start_sphere];

			Real radius = sp.radius;
			Coord center = sp.center;

			if (center.x + radius >= hi.x)
			{
				nbq[cnt + start_i].bodyId1 = pId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(-1, 0, 0);
				nbq[cnt + start_i].pos1 = center + Coord(radius, 0, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = center.x + radius - hi.x;
				cnt++;
			}
			if (center.x - radius <= lo.x)
			{
				nbq[cnt + start_i].bodyId1 = pId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(1, 0, 0);
				nbq[cnt + start_i].pos1 = center - Coord(radius, 0, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = lo.x - (center.x - radius);
				cnt++;
			}

			if (center.y + radius >= hi.y)
			{
				nbq[cnt + start_i].bodyId1 = pId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, -1, 0);
				nbq[cnt + start_i].pos1 = center + Coord(0, radius, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = center.y + radius - hi.y;
				cnt++;
			}
			if (center.y - radius <= lo.y)
			{
				nbq[cnt + start_i].bodyId1 = pId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, 1, 0);
				nbq[cnt + start_i].pos1 = center - Coord(0, radius, 0);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = lo.y - (center.y - radius);
				cnt++;
			}

			if (center.z + radius >= hi.z)
			{
				nbq[cnt + start_i].bodyId1 = pId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, 0, -1);
				nbq[cnt + start_i].pos1 = center + Coord(0, 0, radius);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = center.z + radius - hi.z;
				cnt++;
			}
			if (center.z - radius <= lo.z)
			{
				nbq[cnt + start_i].bodyId1 = pId;
				nbq[cnt + start_i].bodyId2 = -1;
				nbq[cnt + start_i].normal1 = Coord(0, 0, 1);
				nbq[cnt + start_i].pos1 = center - Coord(0, 0, radius);
				nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
				nbq[cnt + start_i].interpenetration = lo.z - (center.z - radius);
				cnt++;
			}
		}
		else if (pId >= start_box && pId < start_tet)//box
		{
			//int idx = pId - start_box;
			int cnt = 0;
			int start_i = count[pId];
			Coord center = box[pId - start_box].center;
			Coord u = box[pId - start_box].u;
			Coord v = box[pId - start_box].v;
			Coord w = box[pId - start_box].w;
			Coord extent = box[pId - start_box].extent;
			Point3D p[8];
			p[0] = Point3D(center - u * extent[0] - v * extent[1] - w * extent[2]);
			p[1] = Point3D(center - u * extent[0] - v * extent[1] + w * extent[2]);
			p[2] = Point3D(center - u * extent[0] + v * extent[1] - w * extent[2]);
			p[3] = Point3D(center - u * extent[0] + v * extent[1] + w * extent[2]);
			p[4] = Point3D(center + u * extent[0] - v * extent[1] - w * extent[2]);
			p[5] = Point3D(center + u * extent[0] - v * extent[1] + w * extent[2]);
			p[6] = Point3D(center + u * extent[0] + v * extent[1] - w * extent[2]);
			p[7] = Point3D(center + u * extent[0] + v * extent[1] + w * extent[2]);
			bool c1, c2, c3, c4, c5, c6;
			c1 = c2 = c3 = c4 = c5 = c6 = true;
			for (int i = 0; i < 8; i++)
			{
				Coord pos = p[i].origin;
				if (pos[0] > hi[0] && c1)
				{
					c1 = true;
					nbq[cnt + start_i].bodyId1 = pId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(-1,0,0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = pos[0] - hi[0];
					cnt++;
				}
				if (pos[1] > hi[1] && c2)
				{
					c2 = true;
					nbq[cnt + start_i].bodyId1 = pId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, -1, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = pos[1] - hi[1];
					cnt++;
				}
				if (pos[2] > hi[2] && c3)
				{
					c3 = true;
					nbq[cnt + start_i].bodyId1 = pId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 0, -1);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = pos[2] - hi[2];
					cnt++;
				}
				if (pos[0] < lo[0] && c4)
				{
					c4 = true;
					nbq[cnt + start_i].bodyId1 = pId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(1, 0, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = lo[0] - pos[0];
					cnt++;
				}
				if (pos[1] < lo[1] && c5)
				{
					c5 = true;
					nbq[cnt + start_i].bodyId1 = pId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 1, 0);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = lo[1] - pos[1];
					cnt++;
				}
				if (pos[2] < lo[2] && c6)
				{
					c6 = true;
					nbq[cnt + start_i].bodyId1 = pId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 0, 1);
					nbq[cnt + start_i].pos1 = pos;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = lo[2] - pos[2];
					cnt++;
				}

			}

		}
		else if (pId >= start_tet && pId < start_segment) // tets
		{
			//printf("???????\n");
			int cnt = 0;
			int start_i = count[pId];

			Tet3D tet_i = tet[pId - start_tet];

			for(int i = 0; i < 4; i ++)
			{ 
				Coord vertex = tet_i.v[i];
				if (vertex.x >= hi.x)
				{
					nbq[cnt + start_i].bodyId1 = pId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(-1, 0, 0);
					nbq[cnt + start_i].pos1 = vertex;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = vertex.x - hi.x;
					cnt++;
				}
				if (vertex.x <= lo.x)
				{
					nbq[cnt + start_i].bodyId1 = pId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(1, 0, 0);
					nbq[cnt + start_i].pos1 = vertex;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = lo.x - (vertex.x);
					cnt++;
				}

				if (vertex.y >= hi.y)
				{
					nbq[cnt + start_i].bodyId1 = pId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, -1, 0);
					nbq[cnt + start_i].pos1 = vertex;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = vertex.y - hi.y;
					cnt++;
				}
				if (vertex.y <= lo.y)
				{
					nbq[cnt + start_i].bodyId1 = pId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 1, 0);
					nbq[cnt + start_i].pos1 = vertex;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = lo.y - (vertex.y);
					cnt++;
				}

				if (vertex.z >= hi.z)
				{
					nbq[cnt + start_i].bodyId1 = pId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 0, -1);
					nbq[cnt + start_i].pos1 = vertex;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = vertex.z - hi.z;
					cnt++;
				}
				if (vertex.z <= lo.z)
				{
					nbq[cnt + start_i].bodyId1 = pId;
					nbq[cnt + start_i].bodyId2 = -1;
					nbq[cnt + start_i].normal1 = Coord(0, 0, 1);
					nbq[cnt + start_i].pos1 = vertex;
					nbq[cnt + start_i].contactType = ContactType::CT_BOUDNARY;
					nbq[cnt + start_i].interpenetration = lo.z - (vertex.z);
					cnt++;
				}
			}
		}
		else//segments 
		{}
	}

	template <typename ContactPair>
	__global__ void SetupFrictionConstraints(
		DArray<ContactPair> nbq,
		int contact_size)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= contact_size) return;

		Coord3D n = nbq[pId].normal1;
		n /= n.norm();

		Coord3D n1, n2;
		if (abs(n[1]) > EPSILON || abs(n[2]) > EPSILON)
		{
			n1 = Coord3D(0, n[2], -n[1]);
			n1 /= n1.norm();
			n2 = n1.cross(n);
			n2 /= n2.norm();
		}
		else if(abs(n[0]) > EPSILON)
		{
			n1 = Coord3D(n[2], 0, -n[0]);
			n1 /= n1.norm();
			n2 = n1.cross(n);
			n2 /= n2.norm();
		}

		nbq[pId * 2 + contact_size].bodyId1 = nbq[pId].bodyId1;
		nbq[pId * 2 + contact_size].bodyId2 = nbq[pId].bodyId2;
		nbq[pId * 2 + contact_size] = nbq[pId];
		nbq[pId * 2 + contact_size].contactType = ContactType::CT_FRICTION;
		nbq[pId * 2 + contact_size].normal1 = n1;

		nbq[pId * 2 + 1 + contact_size].bodyId1 = nbq[pId].bodyId1;
		nbq[pId * 2 + 1 + contact_size].bodyId2 = nbq[pId].bodyId2;
		nbq[pId * 2 + 1 + contact_size] = nbq[pId];
		nbq[pId * 2 + 1 + contact_size].contactType = ContactType::CT_FRICTION;
		nbq[pId * 2 + 1 + contact_size].normal1 = n2;
	}

	template <typename Coord>
	__global__ void CountContactsWithBoundary(
		DArray<Sphere3D> sphere,
		DArray<Box3D> box,
		DArray<Tet3D> tet,
		DArray<int> count,
		Coord hi,
		Coord lo,
		int start_sphere,
		int start_box,
		int start_tet,
		int start_segment)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= sphere.size() + box.size() + tet.size()) return;

		if (pId < start_box && pId >= start_sphere)//sphere
		{
			int cnt = 0;

			Sphere3D sp = sphere[pId - start_sphere];

			Real radius = sp.radius;
			Coord center = sp.center;
			
			if (center.x + radius >= hi.x)
			{
				cnt++;
			}
			if (center.x - radius <= lo.x)
			{
				cnt++;
			}

			if (center.y + radius >= hi.y)
			{
				cnt++;
			}
			if (center.y - radius <= lo.y)
			{
				cnt++;
			}

			if (center.z + radius >= hi.z)
			{
				cnt++;
			}
			if (center.z - radius <= lo.z)
			{
				cnt++;
			}

			count[pId] = cnt;
		}
		else if (pId >= start_box && pId < start_tet)//box
		{
			//int idx = pId - start_box;
			int cnt = 0;
//				int start_i;
			Coord center = box[pId - start_box].center;
			Coord u = box[pId - start_box].u;
			Coord v = box[pId - start_box].v;
			Coord w = box[pId - start_box].w;
			Coord extent = box[pId - start_box].extent;
			Point3D p[8];
			p[0] = Point3D(center - u * extent[0] - v * extent[1] - w * extent[2]);
			p[1] = Point3D(center - u * extent[0] - v * extent[1] + w * extent[2]);
			p[2] = Point3D(center - u * extent[0] + v * extent[1] - w * extent[2]);
			p[3] = Point3D(center - u * extent[0] + v * extent[1] + w * extent[2]);
			p[4] = Point3D(center + u * extent[0] - v * extent[1] - w * extent[2]);
			p[5] = Point3D(center + u * extent[0] - v * extent[1] + w * extent[2]);
			p[6] = Point3D(center + u * extent[0] + v * extent[1] - w * extent[2]);
			p[7] = Point3D(center + u * extent[0] + v * extent[1] + w * extent[2]);
			bool c1, c2, c3, c4, c5, c6;
			c1 = c2 = c3 = c4 = c5 = c6 = true;
			for (int i = 0; i < 8; i++)
			{
				Coord pos = p[i].origin;
				if (pos[0] > hi[0] && c1)
				{
					c1 = true;
					cnt++;
				}
				if (pos[1] > hi[1] && c2)
				{
					c2 = true;
					cnt++;
				}
				if (pos[2] > hi[2] && c3)
				{
					c3 = true;
					cnt++;
				}
				if (pos[0] < lo[0] && c4)
				{
					c4 = true;
					cnt++;
				}
				if (pos[1] < lo[1] && c5)
				{
					c5 = true;
					cnt++;
				}
				if (pos[2] < lo[2] && c6)
				{
					c6 = true;
					cnt++;
				}
			}
			count[pId] = cnt;
		}
		else if (pId >= start_tet && pId < start_segment) // tets
		{
			int cnt = 0;
			int start_i = count[pId];

			Tet3D tet_i = tet[pId - start_tet];

			for (int i = 0; i < 4; i++)
			{
				Coord vertex = tet_i.v[i];
				if (vertex.x >= hi.x)
				{
					cnt++;
				}
				if (vertex.x <= lo.x)
				{
					cnt++;
				}

				if (vertex.y >= hi.y)
				{
					cnt++;
				}
				if (vertex.y <= lo.y)
				{
					cnt++;
				}

				if (vertex.z >= hi.z)
				{
					cnt++;
				}
				if (vertex.z <= lo.z)
				{
					cnt++;
				}
			}

			count[pId] = cnt;
		}
		else//segments
		{}
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::detectCollisionWithBoundary()
	{
		auto discreteSet = TypeInfo::cast<DiscreteElements<DataType3f>>(this->currentTopology()->getDataPtr());
		uint totalSize = discreteSet->totalSize();

		ElementOffset offset = discreteSet->calculateElementOffset();

		int sum = 0;

		mBoundaryContactCounter.resize(discreteSet->totalSize());
		mBoundaryContactCounter.reset();
		if (discreteSet->totalSize() > 0)
		{
			cuExecute(totalSize,
				CountContactsWithBoundary,
				discreteSet->getSpheres(),
				discreteSet->getBoxes(),
				discreteSet->getTets(),
				mBoundaryContactCounter,
				mUpperCorner,
				mLowerCorner,
				0,
				offset.boxIndex(),
				offset.tetIndex(),
				offset.capsuleIndex());

			sum += m_reduce.accumulate(mBoundaryContactCounter.begin(), mBoundaryContactCounter.size());
			m_scan.exclusive(mBoundaryContactCounter, true);

			mBoundaryContacts.resize(sum);

			if (sum > 0) {
				cuExecute(totalSize,
					SetupContactsWithBoundary,
					discreteSet->getSpheres(),
					discreteSet->getBoxes(),
					discreteSet->getTets(),
					mBoundaryContactCounter,
					mBoundaryContacts,
					mUpperCorner,
					mLowerCorner,
					0,
					offset.boxIndex(),
					offset.tetIndex(),
					offset.capsuleIndex());
			}
		}
		else
			mBoundaryContacts.resize(0);
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::initializeJacobian(Real dt)
	{
		auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->currentTopology()->getDataPtr());

		mElementQuery->update();

		auto& contacts = mElementQuery->outContacts()->getData();

		detectCollisionWithBoundary();

		int sizeOfContacts = mBoundaryContacts.size() + contacts.size();

		int sizeOfConstraints = sizeOfContacts;
		if (this->varFrictionEnabled()->getData())
		{
			sizeOfConstraints += 2 * sizeOfContacts;
		}

		mAllConstraints.resize(sizeOfConstraints);
		
		mJ.resize(4 * sizeOfConstraints);
		mB.resize(4 * sizeOfConstraints);
		mAccel.resize(currentCenter()->getElementCount() * 2);
		mD.resize(sizeOfConstraints);
		mEta.resize(sizeOfConstraints);
		mLambda.resize(sizeOfConstraints);

		mJ.reset();
		mB.reset();
		mD.reset();
		mEta.reset();
		mAccel.reset();
		mLambda.reset();
		nbrContacts.reset();

		if (sizeOfConstraints == 0) return;

		if (contacts.size() > 0)
			mAllConstraints.assign(contacts, contacts.size());
		
		if (mBoundaryContacts.size() > 0)
			mAllConstraints.assign(mBoundaryContacts, mBoundaryContacts.size(), contacts.size(), 0);

		if (this->varFrictionEnabled()->getData())
		{
			cuExecute(sizeOfContacts, 
				SetupFrictionConstraints,
				mAllConstraints,
				sizeOfContacts);
		}
		cuExecute(sizeOfConstraints,
			CalculateNbrCons,
			mAllConstraints,
			nbrContacts
		);
		cuExecute(sizeOfConstraints,
			CalculateJacobians,
			mJ,
			mB,
			this->currentCenter()->getData(),
			this->currentInertia()->getData(),
			this->currentMass()->getData(),
			mAllConstraints);

		cuExecute(sizeOfConstraints,
			CalculateDiagonals,
			mD,
			mJ,
			mB);

		cuExecute(sizeOfConstraints, 
			CalculateEta,
			mEta,
			this->currentVelocity()->getData(),
			this->currentAngularVelocity()->getData(),
			mJ,
			this->currentMass()->getData(),
			mAllConstraints,
			dt);
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::updateStates()
	{
		Real dt = this->varTimeStep()->getData();
		//construct j
		initializeJacobian(dt);

		int size_constraints = mAllConstraints.size();
		for (int i = 0; i < 35; i++)
		{
			cuExecute(size_constraints,
				TakeOneJacobiIteration,
				mLambda,
				mAccel,
				mD,
				mJ,
				mB,
				mEta,
				currentMass()->getData(),
				mAllConstraints,
				nbrContacts);
		}

		uint num = this->currentCenter()->getElementCount();
		cuExecute(num,
			RB_UpdateVelocity,
			this->currentVelocity()->getData(),
			this->currentAngularVelocity()->getData(),
			mAccel,
			dt);

		cuExecute(num,
			RB_UpdateGesture,
			this->currentCenter()->getData(),
			this->currentQuaternion()->getData(),
			this->currentRotationMatrix()->getData(),
			this->currentInertia()->getData(),
			this->currentVelocity()->getData(),
			this->currentAngularVelocity()->getData(),
			mInitialInertia,
			dt);
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::updateTopology()
	{
		auto discreteSet = TypeInfo::cast<DiscreteElements<DataType3f>>(this->currentTopology()->getDataPtr());

		ElementOffset offset = discreteSet->calculateElementOffset();

		cuExecute(mDeviceBoxes.size(),
			UpdateBoxes,
			discreteSet->getBoxes(),
			mDeviceBoxes,
			this->currentCenter()->getData(),
			this->currentRotationMatrix()->getData(),
			offset.boxIndex());

		cuExecute(mDeviceBoxes.size(),
			UpdateSpheres,
			discreteSet->getSpheres(),
			this->currentCenter()->getData(),
			offset.sphereIndex());

		cuExecute(mDeviceTets.size(),
			UpdateTets,
			discreteSet->getTets(),
			mDeviceTets,
			this->currentCenter()->getData(),
			this->currentRotationMatrix()->getData(),
			offset.tetIndex());
	}

	DEFINE_CLASS(RigidBodySystem);
}