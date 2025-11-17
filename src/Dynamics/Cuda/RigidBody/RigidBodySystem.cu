#include "Collision/CollisionData.h"
#include "Platform.h"
#include "RigidBodySystem.h"

#include "Primitive/Primitive3D.h"
#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionBoundingBox.h"

#include "RigidBody/Module/TJConstraintSolver.h"
#include "RigidBody/Module/TJSConstraintSolver.h"
#include "RigidBody/Module/TJSoftConstraintSolver.h"
#include "RigidBody/Module/PJSNJSConstraintSolver.h"
//#include "RigidBody/Module/PJSConstraintSolver.h"
#include "RigidBody/Module/PJSoftConstraintSolver.h"
#include "RigidBody/Module/PCGConstraintSolver.h"
#include "RigidBody/Module/CarDriver.h"

//Module headers
#include "RigidBody/Module/ContactsUnion.h"
#include "Topology/DiscreteElements.h"
#include <memory>

namespace dyno
{
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
		this->stateAttribute()->connect(elementQuery->inAttribute());
		this->animationPipeline()->pushModule(elementQuery);
		//elementQuery->varSelfCollision()->setValue(false);

		auto cdBV = std::make_shared<CollistionDetectionBoundingBox<TDataType>>();
		this->stateTopology()->connect(cdBV->inDiscreteElements());
		this->animationPipeline()->pushModule(cdBV);

		auto merge = std::make_shared<ContactsUnion<TDataType>>();
		elementQuery->outContacts()->connect(merge->inContactsA());
		cdBV->outContacts()->connect(merge->inContactsB());

		this->animationPipeline()->pushModule(merge);

		auto iterSolver = std::make_shared<TJSConstraintSolver<TDataType>>();
		this->stateTimeStep()->connect(iterSolver->inTimeStep());
		this->varFrictionEnabled()->connect(iterSolver->varFrictionEnabled());
		this->varGravityEnabled()->connect(iterSolver->varGravityEnabled());
		this->varGravityValue()->connect(iterSolver->varGravityValue());
		this->varFrictionCoefficient()->connect(iterSolver->varFrictionCoefficient());
		this->varSlop()->connect(iterSolver->varSlop());
		this->stateMass()->connect(iterSolver->inMass());
		
		this->stateFrictionCoefficients()->connect(iterSolver->inFrictionCoefficients());
		this->stateAttribute()->connect(iterSolver->inAttribute());
		this->stateCenter()->connect(iterSolver->inCenter());
		this->stateVelocity()->connect(iterSolver->inVelocity());
		this->stateAngularVelocity()->connect(iterSolver->inAngularVelocity());
		this->stateRotationMatrix()->connect(iterSolver->inRotationMatrix());
		this->stateInertia()->connect(iterSolver->inInertia());
		this->stateQuaternion()->connect(iterSolver->inQuaternion());
		this->stateInitialInertia()->connect(iterSolver->inInitialInertia());
		this->stateTopology()->connect(iterSolver->inDiscreteElements());
		merge->outContacts()->connect(iterSolver->inContacts());
		this->animationPipeline()->pushModule(iterSolver);


		this->setDt(0.016f);
	}

	template<typename TDataType>
	RigidBodySystem<TDataType>::~RigidBodySystem()
	{
	}

	template<typename Real>
	SquareMatrix<Real, 3> ParallelAxisTheorem(Vector<Real, 3> offset, Real m)
	{
		SquareMatrix<Real, 3> mat;
		mat(0, 0) = m * (offset.y * offset.y + offset.z * offset.z);
		mat(1, 1) = m * (offset.x * offset.x + offset.z * offset.z);
		mat(2, 2) = m * (offset.x * offset.x + offset.y * offset.y);

		mat(0, 1) = m * offset.x * offset.y;
		mat(1, 0) = m * offset.x * offset.y;

		mat(0, 2) = m * offset.x * offset.z;
		mat(2, 0) = m * offset.z * offset.x;

		mat(1, 2) = m * offset.y * offset.z;
		mat(2, 1) = m * offset.z * offset.y;

		return mat;
	}

	template<typename TDataType>
	std::shared_ptr<PdActor> RigidBodySystem<TDataType>::addBox(
		const BoxInfo& box,
		const RigidBodyInfo& bodyDef, 
		const Real density)
	{
		auto actor = this->createRigidBody(bodyDef);
		this->bindBox(actor, box, density);

		return actor;
	}

	template<typename TDataType>
	std::shared_ptr<PdActor> RigidBodySystem<TDataType>::addSphere(
		const SphereInfo& sphere, 
		const RigidBodyInfo& bodyDef,
		const Real density /*= Real(1)*/)
	{
		auto actor = this->createRigidBody(bodyDef);
		this->bindSphere(actor, sphere, density);

		return actor;
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
	std::shared_ptr<dyno::PdActor> RigidBodySystem<TDataType>::createRigidBody(
		const Coord& p, 
		const TQuat& q)
	{
		return createRigidBody(RigidBodyInfo(p, q));
	}

	template<typename TDataType>
	std::shared_ptr<PdActor> RigidBodySystem<TDataType>::createRigidBody(const RigidBodyInfo& bodyDef, bool isIntial)
	{
		auto bd = bodyDef;
		if (isIntial)
		{
			bd.mass = 0.0f;
			bd.inertia = Mat3f(0.0);
		}
		bd.shapeType = ET_COMPOUND;

		mHostRigidBodyStates.push_back(bd);

		std::shared_ptr<PdActor> actor = std::make_shared<PdActor>();
		actor->idx = mHostRigidBodyStates.size() - 1;
		actor->shapeType = ET_COMPOUND;
		actor->center = bd.position;
		actor->rot = bd.angle;

		return actor;
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::bindBox(
		const std::shared_ptr<PdActor> actor, 
		const BoxInfo& box,
		const Real density /*= Real(100)*/)
	{
		auto& rigidbody = mHostRigidBodyStates[actor->idx];

		float lx = 2.0f * box.halfLength[0];
		float ly = 2.0f * box.halfLength[1];
		float lz = 2.0f * box.halfLength[2];

		Real mass = density * lx * ly * lz;
		printf("%lf\n", mass);

		// Calculate the inertia of box in the local frame
		auto localInertia = 1.0f / 12.0f * mass
			* Mat3f(ly * ly + lz * lz, 0, 0,
				0, lx * lx + lz * lz, 0,
				0, 0, lx * lx + ly * ly);

		// Transform into the rigid body frame
		auto rotShape = box.rot.toMatrix3x3();
		auto rotBody = rigidbody.angle.toMatrix3x3();

		auto rigidbodyInertia = rotBody * (rotShape * localInertia * rotShape.transpose() + ParallelAxisTheorem(box.center, mass)) * rotBody.transpose();
		
		rigidbody.mass += mass;
		rigidbody.inertia += rigidbodyInertia;
		rigidbody.shapeType = ET_COMPOUND;

		mHostShape2RigidBodyMapping.insert(mHostShape2RigidBodyMapping.begin() + mHostSpheres.size() + mHostBoxes.size(), Pair<uint, uint>(mHostBoxes.size(), (uint)actor->idx));
		mHostBoxes.push_back(box);
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::bindSphere(
		const std::shared_ptr<PdActor> actor, 
		const SphereInfo& sphere, 
		const Real density /*= Real(100)*/)
	{
		auto& rigidbody = mHostRigidBodyStates[actor->idx];

		float r = sphere.radius;
		Real mass = 4 / 3.0f * M_PI * r * r * r * density;

		float I11 = r * r;

		// Calculate the inertia of sphere in the local frame
		auto localInertia = 0.4f * mass
			* Mat3f(I11, 0, 0,
				0, I11, 0,
				0, 0, I11);

		auto rotShape = sphere.rot.toMatrix3x3();
		auto rotBody = rigidbody.angle.toMatrix3x3();

		auto rigidbodyInertia = rotBody * (rotShape * localInertia * rotShape.transpose() + ParallelAxisTheorem(sphere.center, mass)) * rotBody.transpose();

		rigidbody.mass += mass;
		rigidbody.inertia += rigidbodyInertia;
		rigidbody.shapeType = ET_COMPOUND;

		mHostShape2RigidBodyMapping.insert(mHostShape2RigidBodyMapping.begin() + mHostSpheres.size(), Pair<uint, uint>(mHostSpheres.size(), (uint)actor->idx));
		mHostSpheres.push_back(sphere);
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::bindCapsule(
		const std::shared_ptr<PdActor> actor, 
		const CapsuleInfo& capsule,
		const Real density /*= Real(100)*/)
	{
		auto& rigidbody = mHostRigidBodyStates[actor->idx];

		Real r = capsule.radius;
		Real h = capsule.halfLength * 2;


		Real mass_hemisphere = 2.0 / 3.0 * M_PI * r * r * r * density;
		Real mass_cylinder = M_PI * r * r * h * density;

		Real I_1_cylinder = 1.0 / 12.0 * mass_cylinder * (3 * r * r + h * h);
		Real I_2_cylinder = 1.0 / 2.0 * mass_cylinder * r * r;


		Real tmp = h / 2 + 3.0 / 8.0 * r;
		Real I_1_hemisphere = mass_hemisphere * (2.0 / 5.0 * r * r + h * h / 2 + 3 * h * r / 8.0);
		Real I_2_hemisphere = 2.0 / 5.0 * mass_hemisphere * r * r;

		Real mass = mass_hemisphere * 2 + mass_cylinder;

		auto localInertia = Mat3f(I_1_cylinder + 2 * I_1_hemisphere, 0, 0,
			0, I_1_cylinder + 2 * I_1_hemisphere, 0,
			0, 0, I_2_cylinder + 2 * I_2_hemisphere);

		auto rotShape = capsule.rot.toMatrix3x3();
		auto rotBody = rigidbody.angle.toMatrix3x3();

		auto rigidbodyInertia = rotBody * (rotShape * localInertia * rotShape.transpose() + ParallelAxisTheorem(capsule.center, mass)) * rotBody.transpose();

		rigidbody.mass += mass;
		rigidbody.inertia += rigidbodyInertia;
		rigidbody.shapeType = ET_COMPOUND;

		mHostShape2RigidBodyMapping.insert(mHostShape2RigidBodyMapping.begin() + mHostSpheres.size() + mHostBoxes.size() + mHostTets.size() + mHostCapsules.size(), Pair<uint, uint>(mHostCapsules.size(), (uint)actor->idx));
		mHostCapsules.push_back(capsule);
	}
	
	template<typename TDataType>
	void RigidBodySystem<TDataType>::bindMedialCone(const std::shared_ptr<PdActor> actor, const MedialConeInfo& medialcone, const Real density)
	{
		auto& rigidbody = mHostRigidBodyStates[actor->idx];

		MedialCone3D cone(medialcone.v[0], medialcone.v[1], medialcone.radius[0], medialcone.radius[1]);

		Real mass = cone.volume() * density;

		float r = medialcone.radius[0] + medialcone.radius[1];

		float I11 = r * r;

		Coord centroid = (cone.v[0] + cone.v[1]) / 2.0;

		// Calculate the inertia of sphere in the local frame
		auto localInertia = 0.4f * mass
			* Mat3f(I11, 0, 0,
				0, I11, 0,
				0, 0, I11);

		auto rotShape = Matrix::identityMatrix();
		auto rotBody = rigidbody.angle.toMatrix3x3();

		auto rigidbodyInertia = rotBody * (rotShape * localInertia * rotShape.transpose() + ParallelAxisTheorem(centroid, mass)) * rotBody.transpose();

		//rigidbody.mass += mass;
		//rigidbody.inertia += rigidbodyInertia;
		rigidbody.shapeType = ET_COMPOUND;

		mHostShape2RigidBodyMapping.insert(mHostShape2RigidBodyMapping.begin() + mHostSpheres.size() + mHostBoxes.size() + mHostTets.size() + mHostCapsules.size() + mHostMedialCones.size(), Pair<uint, uint>(mHostMedialCones.size(), (uint)actor->idx));
		mHostMedialCones.push_back(medialcone);
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::bindMedialSlab(const std::shared_ptr<PdActor> actor, const MedialSlabInfo& medialslab, const Real density)
	{
		auto& rigidbody = mHostRigidBodyStates[actor->idx];

		MedialSlab3D slab(medialslab.v[0], medialslab.v[1], medialslab.v[2], medialslab.radius[0], medialslab.radius[1], medialslab.radius[2]);

		Real mass = slab.volume() * density;

		float r = medialslab.radius[0] + medialslab.radius[1] + medialslab.radius[2];

		float I11 = r * r;

		Coord centroid = (slab.v[0] + slab.v[1] + slab.v[2]) / 3.0;

		// Calculate the inertia of sphere in the local frame
		auto localInertia = 0.4f * mass
			* Mat3f(I11, 0, 0,
				0, I11, 0,
				0, 0, I11);

		auto rotShape = Matrix::identityMatrix();
		auto rotBody = rigidbody.angle.toMatrix3x3();

		auto rigidbodyInertia = rotBody * (rotShape * localInertia * rotShape.transpose() + ParallelAxisTheorem(centroid, mass)) * rotBody.transpose();

		//rigidbody.mass += mass;
		//rigidbody.inertia += rigidbodyInertia;
		rigidbody.shapeType = ET_COMPOUND;

		mHostShape2RigidBodyMapping.insert(mHostShape2RigidBodyMapping.begin() + mHostSpheres.size() + mHostBoxes.size() + mHostTets.size() + mHostCapsules.size() + mHostMedialCones.size() + mHostMedialSlabs.size(), Pair<uint, uint>(mHostMedialSlabs.size(), (uint)actor->idx));
		mHostMedialSlabs.push_back(medialslab);
	}


	template<typename TDataType>
	void RigidBodySystem<TDataType>::bindTet(const std::shared_ptr<PdActor> actor, const TetInfo& tet, const Real density /*= Real(100)*/)
	{
		auto& rigidbody = mHostRigidBodyStates[actor->idx];

		Coord centroid = (tet.v[0] + tet.v[1] + tet.v[2] + tet.v[3]) / 4;

		Coord v0 = tet.v[0] - centroid;
		Coord v1 = tet.v[1] - centroid;
		Coord v2 = tet.v[2] - centroid;
		Coord v3 = tet.v[3] - centroid;

		auto tmpMat = Mat3f(v1 - v0, v2 - v0, v3 - v0);

		Real detJ = abs(tmpMat.determinant());
		Real volume = (1.0 / 6.0) * detJ;
		Real mass = volume * density;

		Real a = density * detJ * (v0.y * v0.y + v0.y * v1.y + v1.y * v1.y + v0.y * v2.y + v1.y * v2.y + v2.y * v2.y + v0.y * v3.y + v1.y * v3.y + v2.y * v3.y + v3.y * v3.y + v0.z * v0.z + v0.z * v1.z + v1.z * v1.z + v0.z * v2.z + v1.z * v2.z + v2.z * v2.z + v0.z * v3.z + v1.z * v3.z + v2.z * v3.z + v3.z * v3.z) / 60;
		Real b = density * detJ * (v0.x * v0.x + v0.x * v1.x + v1.x * v1.x + v0.x * v2.x + v1.x * v2.x + v2.x * v2.x + v0.x * v3.x + v1.x * v3.x + v2.x * v3.x + v3.x * v3.x + v0.z * v0.z + v0.z * v1.z + v1.z * v1.z + v0.z * v2.z + v1.z * v2.z + v2.z * v2.z + v0.z * v3.z + v1.z * v3.z + v2.z * v3.z + v3.z * v3.z) / 60;
		Real c = density * detJ * (v0.x * v0.x + v0.x * v1.x + v1.x * v1.x + v0.x * v2.x + v1.x * v2.x + v2.x * v2.x + v0.x * v3.x + v1.x * v3.x + v2.x * v3.x + v3.x * v3.x + v0.y * v0.y + v0.y * v1.y + v1.y * v1.y + v0.y * v2.y + v1.y * v2.y + v2.y * v2.y + v0.y * v3.y + v1.y * v3.y + v2.y * v3.y + v3.y * v3.y) / 60;
		Real a_ = density * detJ * (2 * v0.y * v0.z + v1.y * v0.z + v2.y * v0.z + v3.y * v0.z + v0.y * v1.z + 2 * v1.y * v1.z + v2.y * v1.z + v3.y * v1.z + v0.y * v2.z + v1.y * v2.z + 2 * v2.y * v2.z + v3.y * v2.z + v0.y * v3.z + v1.y * v3.z + v2.y * v3.z + 2 * v3.y * v3.z) / 120;
		Real b_ = density * detJ * (2 * v0.x * v0.z + v1.x * v0.z + v2.x * v0.z + v3.x * v0.z + v0.x * v1.z + 2 * v1.x * v1.z + v2.x * v1.z + v3.x * v1.z + v0.x * v2.z + v1.x * v2.z + 2 * v2.x * v2.z + v3.x * v2.z + v0.x * v3.z + v1.x * v3.z + v2.x * v3.z + 2 * v3.x * v3.z) / 120;
		Real c_ = density * detJ * (2 * v0.x * v0.y + v1.x * v0.y + v2.x * v0.y + v3.x * v0.y + v0.x * v1.y + 2 * v1.x * v1.y + v2.x * v1.y + v3.x * v1.y + v0.x * v2.y + v1.x * v2.y + 2 * v2.x * v2.y + v3.x * v2.y + v0.x * v3.y + tet.v[1].x * v3.y + v2.x * v3.y + 2 * v3.x * v3.y) / 120;
		Mat3f localInertia(a, -b_, -c_, -b_, b, -a_, -c_, -a_, c);

		auto rotShape = Matrix::identityMatrix();
		auto rotBody = rigidbody.angle.toMatrix3x3();

		auto rigidbodyInertia = rotBody * (rotShape * localInertia * rotShape.transpose() + ParallelAxisTheorem(centroid, mass)) * rotBody.transpose();

		rigidbody.mass += mass;
		rigidbody.inertia += rigidbodyInertia;
		rigidbody.shapeType = ET_COMPOUND;

		mHostShape2RigidBodyMapping.insert(mHostShape2RigidBodyMapping.begin() + mHostSpheres.size() + mHostBoxes.size() + mHostTets.size(), Pair<uint, uint>(mHostTets.size(), (uint)actor->idx));
		mHostTets.push_back(tet);
	}



	template<typename TDataType>
	std::shared_ptr<PdActor> RigidBodySystem<TDataType>::addTet(
		const TetInfo& tetInfo,
		const RigidBodyInfo& bodyDef,
		const Real density /*= Real(1)*/)
	{
		auto actor = this->createRigidBody(bodyDef);
		this->bindTet(actor, tetInfo, density);

		return actor;
	}

	template<typename TDataType>
	std::shared_ptr<PdActor> RigidBodySystem<TDataType>::addCapsule(
		const CapsuleInfo& capsule,
		const RigidBodyInfo& bodyDef, 
		const Real density /*= Real(100)*/)
	{
		auto actor = this->createRigidBody(bodyDef);
		this->bindCapsule(actor, capsule, density);

		return actor;
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
		DArray<Attribute> atts,
		DArray<RigidBodyInfo> states,
		DArray<Real> fricCoeffs,
		ElementOffset offset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= states.size())
			return;
		
		mass[tId] = states[tId].mass;
		rotation[tId] = states[tId].angle.toMatrix3x3();
		velocity[tId] = states[tId].linearVelocity;
		angularVelocity[tId] = rotation[tId] * states[tId].angularVelocity;
		rotation_q[tId] = states[tId].angle;
		pos[tId] = states[tId].position;
		inertia[tId] = states[tId].inertia;
		mask[tId] = states[tId].collisionMask;
		fricCoeffs[tId] = states[tId].friction;

		Attribute att_i;
		att_i.setObjectId(states[tId].bodyId);
		if (states[tId].motionType == BodyType::Static)
		{
			att_i.setFixed();
		}
		else if (states[tId].motionType == BodyType::Kinematic)
		{
			att_i.setPassive();
		}
		else if (states[tId].motionType == BodyType::Dynamic)
		{
			att_i.setDynamic();
		}
		atts[tId] = att_i;
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

	__global__ void SetupCones(
		DArray<MedialCone3D> cone3d,
		DArray<MedialConeInfo> coneInfo
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= coneInfo.size()) return;
		cone3d[tId].v[0] = coneInfo[tId].v[0];
		cone3d[tId].v[1] = coneInfo[tId].v[1];
		cone3d[tId].radius[0] = coneInfo[tId].radius[0];
		cone3d[tId].radius[1] = coneInfo[tId].radius[1];
	}

	__global__ void SetupSlabs(
		DArray<MedialSlab3D> slab3d,
		DArray<MedialSlabInfo> slabInfo
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= slabInfo.size()) return;
		slab3d[tId].v[0] = slabInfo[tId].v[0];
		slab3d[tId].v[1] = slabInfo[tId].v[1];
		slab3d[tId].v[2] = slabInfo[tId].v[2];
		slab3d[tId].radius[0] = slabInfo[tId].radius[0];
		slab3d[tId].radius[1] = slabInfo[tId].radius[1];
		slab3d[tId].radius[2] = slabInfo[tId].radius[2];
	}

	__global__ void SetupCaps(
		DArray<Capsule3D> cap3d,
		DArray<CapsuleInfo> capInfo)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= capInfo.size()) return;

		float halfLenght = capInfo[tId].halfLength;
		Mat3f rot = capInfo[tId].rot.toMatrix3x3();

		cap3d[tId].center = capInfo[tId].center;
		cap3d[tId].rotation = capInfo[tId].rot;
		cap3d[tId].radius = capInfo[tId].radius;
		cap3d[tId].halfLength = capInfo[tId].halfLength;
	}


	template<typename Joint>
	__global__ void UpdateJointIndices(
		DArray<Joint> joints,
		ElementOffset offset)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= joints.size()) return;

		Joint joint = joints[tId];

		joint.bodyId1 += offset.checkElementOffset(joint.bodyType1);
		joint.bodyId2 += offset.checkElementOffset(joint.bodyType2);

		joints[tId] = joint;
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::resetStates()
	{
		auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->stateTopology()->getDataPtr());

		mDeviceBoxes.assign(mHostBoxes);
		mDeviceSpheres.assign(mHostSpheres);
		mDeviceTets.assign(mHostTets);
		mDeviceCapsules.assign(mHostCapsules);
		mDeviceMedialCones.assign(mHostMedialCones);
		mDeviceMedialSlabs.assign(mHostMedialSlabs);

		auto& boxes = topo->boxesInLocal();
		auto& spheres = topo->spheresInLocal();
		auto& tets = topo->tetsInLocal();
		auto& caps = topo->capsulesInLocal();
		auto& cones = topo->medialConesInLocal();
		auto& slabs = topo->medialSlabsInLocal();
		

		boxes.resize(mDeviceBoxes.size());
		spheres.resize(mDeviceSpheres.size());
		tets.resize(mDeviceTets.size());
		caps.resize(mDeviceCapsules.size());
		cones.resize(mDeviceMedialCones.size());
		slabs.resize(mDeviceMedialSlabs.size());

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

		cuExecute(mDeviceMedialCones.size(),
			SetupCones,
			cones,
			mDeviceMedialCones);

		cuExecute(mDeviceMedialSlabs.size(),
			SetupSlabs,
			slabs,
			mDeviceMedialSlabs);

		

		mDeviceRigidBodyStates.assign(mHostRigidBodyStates);

		int sizeOfRigidBodies = mDeviceRigidBodyStates.size();// topo->totalSize();

		ElementOffset eleOffset = topo->calculateElementOffset();

		this->stateRotationMatrix()->resize(sizeOfRigidBodies);
		this->stateAngularVelocity()->resize(sizeOfRigidBodies);
		this->stateCenter()->resize(sizeOfRigidBodies);
		this->stateVelocity()->resize(sizeOfRigidBodies);
		this->stateMass()->resize(sizeOfRigidBodies);
		this->stateInertia()->resize(sizeOfRigidBodies);
		this->stateQuaternion()->resize(sizeOfRigidBodies);
		this->stateCollisionMask()->resize(sizeOfRigidBodies);
		this->stateAttribute()->resize(sizeOfRigidBodies);
		this->stateFrictionCoefficients()->resize(sizeOfRigidBodies);

		cuExecute(sizeOfRigidBodies,
			RB_SetupInitialStates,
			this->stateMass()->getData(),
			this->stateCenter()->getData(),
			this->stateRotationMatrix()->getData(),
			this->stateVelocity()->getData(),
			this->stateAngularVelocity()->getData(),
			this->stateQuaternion()->getData(),
			this->stateInertia()->getData(),
			this->stateCollisionMask()->getData(),
			this->stateAttribute()->getData(),
			mDeviceRigidBodyStates,
			this->stateFrictionCoefficients()->getData(),
			eleOffset);

		this->stateInitialInertia()->resize(sizeOfRigidBodies);
		this->stateInitialInertia()->getDataPtr()->assign(this->stateInertia()->getData());

		topo->ballAndSocketJoints().assign(mHostJointsBallAndSocket);
		topo->sliderJoints().assign(mHostJointsSlider);
		topo->hingeJoints().assign(mHostJointsHinge);
		topo->fixedJoints().assign(mHostJointsFixed);
		topo->pointJoints().assign(mHostJointsPoint);

		uint os = eleOffset.checkElementOffset(ET_CAPSULE);

		cuExecute(topo->ballAndSocketJoints().size(),
			UpdateJointIndices,
			topo->ballAndSocketJoints(),
			eleOffset);

		cuExecute(topo->sliderJoints().size(),
			UpdateJointIndices,
			topo->sliderJoints(),
			eleOffset);

		cuExecute(topo->hingeJoints().size(),
			UpdateJointIndices,
			topo->hingeJoints(),
			eleOffset);

		cuExecute(topo->fixedJoints().size(),
			UpdateJointIndices,
			topo->fixedJoints(),
			eleOffset);

		cuExecute(topo->pointJoints().size(),
			UpdateJointIndices,
			topo->pointJoints(),
			eleOffset);

		setupShape2RigidBodyMapping();

		topo->setPosition(this->stateCenter()->constData());
		topo->setRotation(this->stateRotationMatrix()->constData());
		topo->update();
	}
	
	template<typename TDataType>
	void RigidBodySystem<TDataType>::postUpdateStates()
	{
		auto discreteSet = TypeInfo::cast<DiscreteElements<DataType3f>>(this->stateTopology()->getDataPtr());

		ElementOffset offset = discreteSet->calculateElementOffset();

		if (this->stateCenter()->size() <= 0)
		{
			return;
		}

		discreteSet->setPosition(this->stateCenter()->constData());
		discreteSet->setRotation(this->stateRotationMatrix()->constData());
		discreteSet->update();
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::clearRigidBodySystem()
	{
		mHostRigidBodyStates.clear();

		mHostSpheres.clear();
		mHostBoxes.clear();
		mHostTets.clear();
		mHostCapsules.clear();
		mHostMedialCones.clear();
		mHostMedialSlabs.clear();

		mDeviceRigidBodyStates.clear();

		mDeviceSpheres.clear();
		mDeviceBoxes.clear();
		mDeviceTets.clear();
		mDeviceCapsules.clear();
		mDeviceMedialCones.clear();
		mDeviceMedialSlabs.clear();

		mHostJointsBallAndSocket.clear();
		mHostJointsSlider.clear();
		mHostJointsHinge.clear();
		mHostJointsFixed.clear();
		mHostJointsPoint.clear();
		mHostShape2RigidBodyMapping.clear();

		m_numOfSamples = 0;

		m_deviceSamples.clear();
		m_deviceNormals.clear();

		samples.clear();
		normals.clear();
	}

	__global__ void RBS_ConstructShape2RigidBodyMapping(
		DArray<Pair<uint, uint>> mapping)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mapping.size()) return;

		mapping[pId] = Pair<uint, uint>(pId, pId);
	}

	__global__ void RBS_UpdateShape2RigidBodyMapping(
		DArray<Pair<uint, uint>> mapping,
		ElementOffset offset)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mapping.size()) return;

		Pair<uint, uint> pair = mapping[pId];

		uint first = pair.first;

		if (pId < offset.boxIndex())
		{
		}
		else if (pId < offset.tetIndex())
		{
			first += offset.boxIndex();
		}
		else if (pId < offset.capsuleIndex())
		{
			first += offset.tetIndex();
		}
		else if (pId < offset.triangleIndex())
		{
			first += offset.capsuleIndex();
		}
		else if(pId < offset.medialConeIndex())
		{
			first += offset.triangleIndex();
		}
		else if (pId < offset.medialSlabIndex())
		{
			first += offset.medialConeIndex();
		}
		else
		{
			first += offset.medialSlabIndex();
		}

		mapping[pId] = Pair<uint, uint>(first, pair.second);
	}

	template<typename TDataType>
	void RigidBodySystem<TDataType>::setupShape2RigidBodyMapping()
	{
		auto topo = this->stateTopology()->getDataPtr();
		auto& mapping = topo->shape2RigidBodyMapping();

		uint totalSize = topo->totalSize();

		mapping.assign(mHostShape2RigidBodyMapping);

		cuExecute(totalSize,
			RBS_UpdateShape2RigidBodyMapping,
			mapping,
			topo->calculateElementOffset());
	}

	DEFINE_CLASS(RigidBodySystem);
}