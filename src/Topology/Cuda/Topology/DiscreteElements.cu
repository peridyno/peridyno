#include "DiscreteElements.h"

#include <thrust/sort.h>

namespace dyno
{
	IMPLEMENT_TCLASS(DiscreteElements, TDataType)

	template<typename TDataType>
	DiscreteElements<TDataType>::DiscreteElements()
		: TopologyModule()
	{
	}

	template<typename TDataType>
	DiscreteElements<TDataType>::~DiscreteElements()
	{
// 		m_hostBoxes.clear();
// 		m_hostSpheres.clear();
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::scale(Real s)
	{
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::totalSize()
	{
		return mBoxesInLocal.size() + mSpheresInLocal.size() + mTetsInLocal.size() + mCapsulesInLocal.size() + mTrianglesInLocal.size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::totalJointSize()
	{
		return mBallAndSocketJoints.size() + mSliderJoints.size() + mHingeJoints.size() + mFixedJoints.size() + mPointJoints.size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::sphereIndex()
	{
		return 0;
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::triangleIndex()
	{
		return capsuleIndex() + this->capsulesInLocal().size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::tetIndex()
	{
		return boxIndex() + this->boxesInLocal().size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::capsuleIndex()
	{
		return tetIndex() + this->tetsInLocal().size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::boxIndex()
	{
		return sphereIndex() + this->spheresInLocal().size();
	}

	template<typename TDataType>
	ElementOffset DiscreteElements<TDataType>::calculateElementOffset()
	{
		ElementOffset elementOffset;
		elementOffset.setSphereRange(sphereIndex(), sphereIndex() + this->spheresInLocal().size());
		elementOffset.setBoxRange(boxIndex(), boxIndex() + this->boxesInLocal().size());
		elementOffset.setTetRange(tetIndex(), tetIndex() + this->tetsInLocal().size());
		elementOffset.setCapsuleRange(capsuleIndex(), capsuleIndex() + this->capsulesInLocal().size());
		elementOffset.setTriangleRange(triangleIndex(), triangleIndex() + this->trianglesInLocal().size());

		return elementOffset;
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setBoxes(DArray<Box3D>& boxes)
	{
		mBoxesInLocal.assign(boxes);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setSpheres(DArray<Sphere3D>& spheres)
	{
		mSpheresInLocal.assign(spheres);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTetSDF(DArray<Real>& sdf)
	{
		m_tet_sdf.assign(sdf);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTets(DArray<Tet3D>& tets)
	{
		mTetsInLocal.assign(tets);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setCapsules(DArray<Capsule3D>& capsules)
	{
		mCapsulesInLocal.assign(capsules);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTetBodyId(DArray<int>& body_id)
	{
		m_tet_body_mapping.assign(body_id);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTetElementId(DArray<TopologyModule::Tetrahedron>& element_id)
	{
		m_tet_element_id.assign(element_id);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTriangles(DArray<Triangle3D>& triangles)
	{
		mTrianglesInLocal.assign(triangles);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::copyFrom(DiscreteElements<TDataType>& de)
	{
		mSpheresInLocal.assign(de.mSpheresInLocal);
		mBoxesInLocal.assign(de.mBoxesInLocal);
		mTetsInLocal.assign(de.mTetsInLocal);
		mCapsulesInLocal.assign(de.mCapsulesInLocal);
		mTrianglesInLocal.assign(de.mTrianglesInLocal);

		mSphereInGlobal.assign(de.mSphereInGlobal);
		mBoxInGlobal.assign(de.mBoxInGlobal);
		mTetInGlobal.assign(de.mTetInGlobal);
		mCapsuleInGlobal.assign(de.mCapsuleInGlobal);
		mTriangleInGlobal.assign(de.mTriangleInGlobal);

		mShape2RigidBody.assign(de.mShape2RigidBody);
		mPosition.assign(de.mPosition);
		mRotation.assign(de.mRotation);

		mBallAndSocketJoints.assign(de.mBallAndSocketJoints);
		mSliderJoints.assign(de.mSliderJoints);
		mHingeJoints.assign(de.mHingeJoints);
		mFixedJoints.assign(de.mFixedJoints);
		mPointJoints.assign(de.mPointJoints);
		mDistanceJoints.assign(de.mDistanceJoints);

		m_tet_sdf.assign(de.m_tet_sdf);
		m_tet_body_mapping.assign(de.m_tet_body_mapping);
		m_tet_element_id.assign(de.m_tet_element_id);
	}

	template<typename Joint>
	__global__ void DE_UpdateJointIds(
		DArray<Joint> joints,
		uint size,
		uint offsetJoint,
		uint offsetRigidBody)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= size) return;

		Joint joint = joints[tId + offsetJoint];
		joint.bodyId1 += offsetRigidBody;
		joint.bodyId2 += offsetRigidBody;

		joints[tId + offsetJoint] = joint;
	}

	__global__ void DE_UpdateShape2RigidBodyMapping(
		DArray<Pair<uint, uint>> mapping,
		DArray<uint> offsetShape,	//with a constant array size of 5
		uint size,
		uint offsetMapping,
		uint offsetRigidBody,
		ElementOffset elementOffset)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= size) return;

		Pair<uint, uint> pair = mapping[tId + offsetMapping];

		pair.second += offsetRigidBody;

		//Sphere id
		if (tId < elementOffset.boxIndex())
		{
			pair.first = tId + offsetShape[0];
		}
		//Box id
		else if (tId < elementOffset.tetIndex())
		{
			pair.first = (tId - elementOffset.boxIndex()) + offsetShape[1];
		}
		//Tet id
		else if (tId < elementOffset.capsuleIndex())
		{
			pair.first = (tId - elementOffset.tetIndex()) + offsetShape[2];
		}
		//Capsule id
		else if (tId < elementOffset.triangleIndex())
		{
			pair.first = (tId - elementOffset.tetIndex()) + offsetShape[3];
		}
		else
		{
			pair.first = (tId - elementOffset.triangleIndex()) + offsetShape[4];
		}

		mapping[tId + offsetMapping] = pair;
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::merge(CArray<std::shared_ptr<DiscreteElements<TDataType>>>& topos)
	{
		//Merge shapes
		uint sizeOfSpheres = 0;
		uint sizeOfBoxes = 0;
		uint sizeOfCapsules = 0;
		uint sizeOfTets = 0;
		uint sizeOfTriangles = 0;

		for (uint i = 0; i < topos.size(); i++)
		{
			auto topo = topos[i];

			sizeOfSpheres += topo->spheresInLocal().size();
			sizeOfBoxes += topo->boxesInLocal().size();
			sizeOfCapsules += topo->capsulesInLocal().size();
			sizeOfTets += topo->tetsInLocal().size();
			sizeOfTriangles += topo->trianglesInLocal().size();
		}

		mSpheresInLocal.resize(sizeOfSpheres);
		mBoxesInLocal.resize(sizeOfBoxes);
		mCapsulesInLocal.resize(sizeOfCapsules);
		mTetsInLocal.resize(sizeOfTets);
		mTrianglesInLocal.resize(sizeOfTriangles);

		uint offsetOfSpheres = 0;
		uint offsetOfBoxes = 0;
		uint offsetOfTets = 0;
		uint offsetOfCapsules = 0;
		uint offsetOfTriangles = 0;
		for (uint i = 0; i < topos.size(); i++)
		{
			auto topo = topos[i];

			mSpheresInLocal.assign(topo->spheresInLocal(), topo->spheresInLocal().size(), offsetOfSpheres, 0);
			mBoxesInLocal.assign(topo->boxesInLocal(), topo->boxesInLocal().size(), offsetOfBoxes, 0);
			mCapsulesInLocal.assign(topo->capsulesInLocal(), topo->capsulesInLocal().size(), offsetOfCapsules, 0);
			mTetsInLocal.assign(topo->tetsInLocal(), topo->tetsInLocal().size(), offsetOfTets, 0);
			mTrianglesInLocal.assign(topo->trianglesInLocal(), topo->trianglesInLocal().size(), offsetOfTriangles, 0);

			offsetOfSpheres += topo->spheresInLocal().size();
			offsetOfBoxes += topo->boxesInLocal().size();
			offsetOfCapsules += topo->capsulesInLocal().size();
			offsetOfTets += topo->tetsInLocal().size();
			offsetOfTriangles += topo->trianglesInLocal().size();
		}

		//Merge rigid body states
		uint sizeOfRigidBodies = 0;

		for (uint i = 0; i < topos.size(); i++)
		{
			auto topo = topos[i];

			sizeOfRigidBodies += topo->position().size();
		}

		mPosition.resize(sizeOfRigidBodies);
		mRotation.resize(sizeOfRigidBodies);

		uint offsetOfRigidBodies = 0;
		for (uint i = 0; i < topos.size(); i++)
		{
			auto topo = topos[i];

			mPosition.assign(topo->position(), topo->position().size(), offsetOfRigidBodies, 0);
			mRotation.assign(topo->rotation(), topo->rotation().size(), offsetOfRigidBodies, 0);

			offsetOfRigidBodies += topo->position().size();
		}

		//Merge shape to rigid body mapping
		uint sizeOfMapping = 0;
		for (uint i = 0; i < topos.size(); i++)
		{
			auto topo = topos[i];

			sizeOfMapping += topo->shape2RigidBodyMapping().size();
		}

		mShape2RigidBody.resize(sizeOfMapping);

		uint offsetOfMapping = 0;
		offsetOfRigidBodies = 0;
		CArray<uint> offsetArrayInHost(5);
		offsetArrayInHost[0] = 0;
		offsetArrayInHost[1] = offsetArrayInHost[0] + sizeOfSpheres;
		offsetArrayInHost[2] = offsetArrayInHost[1] + sizeOfBoxes;
		offsetArrayInHost[3] = offsetArrayInHost[2] + sizeOfCapsules;
		offsetArrayInHost[4] = offsetArrayInHost[3] + sizeOfTets;

		DArray<uint> offsetArrayInDevice(5);
		for (uint i = 0; i < topos.size(); i++)
		{
			auto topo = topos[i];

			uint sizeOfShape = topo->shape2RigidBodyMapping().size();

			mShape2RigidBody.assign(topo->shape2RigidBodyMapping(), sizeOfShape, offsetOfMapping, 0);

			offsetArrayInDevice.assign(offsetArrayInHost);

			cuExecute(sizeOfShape, DE_UpdateShape2RigidBodyMapping, mShape2RigidBody, offsetArrayInDevice, sizeOfShape, offsetOfMapping, offsetOfRigidBodies, topo->calculateElementOffset());

			offsetOfMapping += sizeOfShape;
			offsetOfRigidBodies += topo->position().size();

			offsetArrayInHost[0] += topo->spheresInLocal().size();
			offsetArrayInHost[1] += topo->boxesInLocal().size();
			offsetArrayInHost[2] += topo->capsulesInLocal().size();
			offsetArrayInHost[3] += topo->tetsInLocal().size();
			offsetArrayInHost[4] += topo->trianglesInLocal().size();
		}

		thrust::sort(thrust::device, mShape2RigidBody.begin(), mShape2RigidBody.begin() + mShape2RigidBody.size(), thrust::less<Pair<uint, uint>>());

		//Merge joints
		uint sizeOfBallAndSocketJoints = 0;
		uint sizeOfSliderJoints = 0;
		uint sizeOfHingeJoints = 0;
		uint sizeOfFixedJoints = 0;
		uint sizeOfPointJoints = 0;
		uint sizeOfDistanceJoints = 0;

		for (uint i = 0; i < topos.size(); i++)
		{
			auto topo = topos[i];

			sizeOfBallAndSocketJoints += topo->ballAndSocketJoints().size();
			sizeOfSliderJoints += topo->sliderJoints().size();
			sizeOfHingeJoints += topo->hingeJoints().size();
			sizeOfFixedJoints += topo->fixedJoints().size();
			sizeOfPointJoints += topo->pointJoints().size();
			sizeOfDistanceJoints += topo->distanceJoints().size();
		}

		mBallAndSocketJoints.resize(sizeOfBallAndSocketJoints);
		mSliderJoints.resize(sizeOfSliderJoints);
		mHingeJoints.resize(sizeOfHingeJoints);
		mFixedJoints.resize(sizeOfFixedJoints);
		mPointJoints.resize(sizeOfPointJoints);
		mDistanceJoints.resize(sizeOfDistanceJoints);

		uint offsetOfBallAndSocketJoints = 0;
		uint offsetOfSliderJoints = 0;
		uint offsetOfHingeJoints = 0;
		uint offsetOfFixedJoints = 0;
		uint offsetOfPointJoints = 0;
		uint offsetOfDistanceJoints = 0;

		offsetOfRigidBodies = 0;
		for (uint i = 0; i < topos.size(); i++)
		{
			auto topo = topos[i];

			mBallAndSocketJoints.assign(topo->ballAndSocketJoints(), topo->ballAndSocketJoints().size(), offsetOfBallAndSocketJoints, 0);
			mSliderJoints.assign(topo->sliderJoints(), topo->sliderJoints().size(), offsetOfSliderJoints, 0);
			mHingeJoints.assign(topo->hingeJoints(), topo->hingeJoints().size(), offsetOfHingeJoints, 0);
			mFixedJoints.assign(topo->fixedJoints(), topo->fixedJoints().size(), offsetOfFixedJoints, 0);
			mPointJoints.assign(topo->pointJoints(), topo->pointJoints().size(), offsetOfPointJoints, 0);
			mDistanceJoints.assign(topo->distanceJoints(), topo->distanceJoints().size(), offsetOfDistanceJoints, 0);

			cuExecute(topo->ballAndSocketJoints().size(), DE_UpdateJointIds, mBallAndSocketJoints, topo->ballAndSocketJoints().size(), offsetOfBallAndSocketJoints, offsetOfRigidBodies);
			cuExecute(topo->sliderJoints().size(), DE_UpdateJointIds, mSliderJoints, topo->sliderJoints().size(), offsetOfSliderJoints, offsetOfRigidBodies);
			cuExecute(topo->hingeJoints().size(), DE_UpdateJointIds, mHingeJoints, topo->hingeJoints().size(), offsetOfHingeJoints, offsetOfRigidBodies);
			cuExecute(topo->fixedJoints().size(), DE_UpdateJointIds, mFixedJoints, topo->fixedJoints().size(), offsetOfFixedJoints, offsetOfRigidBodies);
			cuExecute(topo->pointJoints().size(), DE_UpdateJointIds, mPointJoints, topo->pointJoints().size(), offsetOfPointJoints, offsetOfRigidBodies);
			cuExecute(topo->distanceJoints().size(), DE_UpdateJointIds, mDistanceJoints, topo->distanceJoints().size(), offsetOfDistanceJoints, offsetOfRigidBodies);

			offsetOfBallAndSocketJoints += topo->ballAndSocketJoints().size();
			offsetOfSliderJoints += topo->sliderJoints().size();
			offsetOfHingeJoints += topo->hingeJoints().size();
			offsetOfFixedJoints += topo->fixedJoints().size();
			offsetOfPointJoints += topo->pointJoints().size();
			offsetOfDistanceJoints += topo->distanceJoints().size();

			offsetOfRigidBodies += topo->position().size();
		}

		this->update();
	}

	// Some useful tools to to do transformation for discrete element

	template<typename Real>
	DYN_FUNC TOrientedBox3D<Real> local2Global(const TOrientedBox3D<Real>& box, const Vector<Real, 3>& t, const SquareMatrix<Real, 3>& r)
	{
		TOrientedBox3D<Real> ret;
		ret.center = t + r * box.center;
		ret.u = r * box.u;
		ret.v = r * box.v;
		ret.w = r * box.w;
		ret.extent = box.extent;

		return ret;
	}

	template<typename Real>
	DYN_FUNC TSphere3D<Real> local2Global(const TSphere3D<Real>& sphere, const Vector<Real, 3>& t, const SquareMatrix<Real, 3>& r)
	{
		TSphere3D<Real> ret;
		ret.center = t + r * sphere.center;
		ret.radius = sphere.radius;
		ret.rotation = Quat<Real>(r * sphere.rotation.toMatrix3x3());

		return ret;
	}

	template<typename Real>
	DYN_FUNC TCapsule3D<Real> local2Global(const TCapsule3D<Real>& capsule, const Vector<Real, 3>& t, const SquareMatrix<Real, 3>& r)
	{
		TCapsule3D<Real> ret;
		ret.center = t + r * capsule.center;
		ret.radius = capsule.radius;
		ret.halfLength = capsule.halfLength;
		ret.rotation = Quat<Real>(r * capsule.rotation.toMatrix3x3());

		return ret;
	}

	template<typename Real>
	DYN_FUNC TTet3D<Real> local2Global(const TTet3D<Real>& tet, const Vector<Real, 3>& t, const SquareMatrix<Real, 3>& r)
	{
		TTet3D<Real> ret;
		ret.v[0] = t + r * tet.v[0];
		ret.v[1] = t + r * tet.v[1];
		ret.v[2] = t + r * tet.v[2];
		ret.v[3] = t + r * tet.v[3];

		return ret;
	}

	template<typename Real>
	DYN_FUNC TTriangle3D<Real> local2Global(const TTriangle3D<Real>& tri, const Vector<Real, 3>& t, const SquareMatrix<Real, 3>& r)
	{
		TTriangle3D<Real> ret;
		ret.v[0] = t + r * tri.v[0];
		ret.v[1] = t + r * tri.v[1];
		ret.v[2] = t + r * tri.v[2];

		return ret;
	}

	template<typename Coord, typename Matrix, typename Box3D>
	__global__ void DE_Local2Global(
		DArray<Box3D> boxInGlobal,
		DArray<Sphere3D> sphereInGlobal,
		DArray<Tet3D> tetInGlobal,
		DArray<Capsule3D> capInGlobal,
		DArray<Triangle3D> triInGlobal,
		DArray<Box3D> boxInLocal,
		DArray<Sphere3D> sphereInLocal,
		DArray<Tet3D> tetInLocal,
		DArray<Capsule3D> capInLocal,
		DArray<Triangle3D> triInLocal,
		DArray<Coord> positionGlobal,
		DArray<Matrix> rotationGlobal,
		DArray<Pair<uint, uint>> mapping,
		ElementOffset elementOffset,
		uint totalSize)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= totalSize) return;

		ElementType eleType = elementOffset.checkElementType(tId);

		uint rigidbodyId = mapping[tId].second;

		Coord t = positionGlobal[rigidbodyId];
		Matrix r = rotationGlobal[rigidbodyId];

		switch (eleType)
		{
		case ET_SPHERE:
		{
			sphereInGlobal[tId - elementOffset.sphereIndex()] = local2Global(sphereInLocal[tId - elementOffset.sphereIndex()], t, r);
			break;
		}
		case ET_BOX:
		{
			boxInGlobal[tId - elementOffset.boxIndex()] = local2Global(boxInLocal[tId - elementOffset.boxIndex()], t, r);
			break;
		}
		case ET_TET:
		{
			tetInGlobal[tId - elementOffset.tetIndex()] = local2Global(tetInLocal[tId - elementOffset.tetIndex()], t, r);
			break;
		}
		case ET_CAPSULE:
		{
			capInGlobal[tId - elementOffset.capsuleIndex()] = local2Global(capInLocal[tId - elementOffset.capsuleIndex()], t, r);
			break;
		}
		case ET_TRI:
		{
			triInGlobal[tId - elementOffset.triangleIndex()] = local2Global(triInLocal[tId - elementOffset.triangleIndex()], t, r);
			break;
		}
		default:
			break;
		}
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::requestDiscreteElementsInGlobal(
		DArray<Box3D>& boxInGlobal, 
		DArray<Sphere3D>& sphereInGlobal, 
		DArray<Tet3D>& tetInGlobal, 
		DArray<Capsule3D>& capInGlobal,
		DArray<Triangle3D>& triInGlobal)
	{
		auto elementOffset = this->calculateElementOffset();

		boxInGlobal.assign(this->boxesInLocal());
		sphereInGlobal.assign(this->spheresInLocal());
		tetInGlobal.assign(this->tetsInLocal());
		capInGlobal.assign(this->capsulesInLocal());
		triInGlobal.assign(this->trianglesInLocal());

		uint num = this->totalSize();

		cuExecute(num,
			DE_Local2Global,
			boxInGlobal,
			sphereInGlobal,
			tetInGlobal,
			capInGlobal,
			triInGlobal,
			this->boxesInLocal(),
			this->spheresInLocal(),
			this->tetsInLocal(),
			this->capsulesInLocal(),
			this->trianglesInLocal(),
			mPosition,
			mRotation,
			mShape2RigidBody,
			elementOffset,
			num);
	}

	template<typename Coord, typename Matrix, typename Box3D>
	__global__ void DE_Local2GlobalForBox(
		DArray<Box3D> boxInGlobal,
		DArray<Box3D> boxInLocal,
		DArray<Coord> positionGlobal,
		DArray<Matrix> rotationGlobal,
		DArray<Pair<uint, uint>> mapping,
		uint offset)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boxInLocal.size()) return;

		uint rigidbodyId = mapping[tId + offset].second;

		Coord t = positionGlobal[rigidbodyId];
		Matrix r = rotationGlobal[rigidbodyId];

		boxInGlobal[tId] = local2Global(boxInLocal[tId], t, r);
	}

	template<typename Coord, typename Matrix, typename Sphere3D>
	__global__ void DE_Local2GlobalForSphere(
		DArray<Sphere3D> sphereInGlobal,
		DArray<Sphere3D> sphereInLocal,
		DArray<Coord> positionGlobal,
		DArray<Matrix> rotationGlobal,
		DArray<Pair<uint, uint>> mapping,
		uint offset)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sphereInLocal.size()) return;

		uint rigidbodyId = mapping[tId + offset].second;

		Coord t = positionGlobal[rigidbodyId];
		Matrix r = rotationGlobal[rigidbodyId];

		sphereInGlobal[tId] = local2Global(sphereInLocal[tId], t, r);
	}

	template<typename Coord, typename Matrix, typename Tet3D>
	__global__ void DE_Local2GlobalForTet(
		DArray<Tet3D> tetInGlobal,
		DArray<Tet3D> tetInLocal,
		DArray<Coord> positionGlobal,
		DArray<Matrix> rotationGlobal,
		DArray<Pair<uint, uint>> mapping,
		uint offset)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tetInLocal.size()) return;

		uint rigidbodyId = mapping[tId + offset].second;

		Coord t = positionGlobal[rigidbodyId];
		Matrix r = rotationGlobal[rigidbodyId];

		tetInGlobal[tId] = local2Global(tetInLocal[tId], t, r);
	}

	template<typename Coord, typename Matrix, typename Capsule3D>
	__global__ void DE_Local2GlobalForCapsule(
		DArray<Capsule3D> capInGlobal,
		DArray<Capsule3D> capInLocal,
		DArray<Coord> positionGlobal,
		DArray<Matrix> rotationGlobal,
		DArray<Pair<uint, uint>> mapping,
		uint offset)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= capInLocal.size()) return;

		uint rigidbodyId = mapping[tId + offset].second;

		Coord t = positionGlobal[rigidbodyId];
		Matrix r = rotationGlobal[rigidbodyId];

		capInGlobal[tId] = local2Global(capInLocal[tId], t, r);
	}

	template<typename Coord, typename Matrix, typename Triangle3D>
	__global__ void DE_Local2GlobalForTriangle(
		DArray<Triangle3D> triInGlobal,
		DArray<Triangle3D> triInLocal,
		DArray<Coord> positionGlobal,
		DArray<Matrix> rotationGlobal,
		DArray<Pair<uint, uint>> mapping,
		uint offset)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triInLocal.size()) return;

		uint rigidbodyId = mapping[tId + offset].second;

		Coord t = positionGlobal[rigidbodyId];
		Matrix r = rotationGlobal[rigidbodyId];

		triInGlobal[tId] = local2Global(triInLocal[tId], t, r);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::requestCapsuleInGlobal(DArray<Capsule3D>& capInGlobal)
	{
		auto elementOffset = this->calculateElementOffset();

		capInGlobal.assign(this->capsulesInLocal());

		cuExecute(capInGlobal.size(),
			DE_Local2GlobalForCapsule,
			capInGlobal,
			this->capsulesInLocal(),
			mPosition,
			mRotation,
			this->shape2RigidBodyMapping(),
			elementOffset.capsuleIndex());
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::requestTriangleInGlobal(DArray<Triangle3D>& triInGlobal)
	{
		auto elementOffset = this->calculateElementOffset();

		triInGlobal.assign(this->trianglesInLocal());
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::requestTetInGlobal(DArray<Tet3D>& tetInGlobal)
	{
		auto elementOffset = this->calculateElementOffset();

		tetInGlobal.assign(this->tetsInLocal());

		cuExecute(tetInGlobal.size(),
			DE_Local2GlobalForCapsule,
			tetInGlobal,
			this->tetsInLocal(),
			mPosition,
			mRotation,
			this->shape2RigidBodyMapping(),
			elementOffset.tetIndex());
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::requestSphereInGlobal(DArray<Sphere3D>& sphereInGlobal)
	{
		auto elementOffset = this->calculateElementOffset();

		sphereInGlobal.assign(this->spheresInLocal());

		cuExecute(sphereInGlobal.size(),
			DE_Local2GlobalForSphere,
			sphereInGlobal,
			this->spheresInLocal(),
			mPosition,
			mRotation,
			this->shape2RigidBodyMapping(),
			elementOffset.sphereIndex());
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::requestBoxInGlobal(DArray<Box3D>& boxInGlobal)
	{
		auto elementOffset = this->calculateElementOffset();

		boxInGlobal.assign(this->boxesInLocal());

		cuExecute(boxInGlobal.size(),
			DE_Local2GlobalForBox,
			boxInGlobal,
			this->boxesInLocal(),
			mPosition,
			mRotation,
			this->shape2RigidBodyMapping(),
			elementOffset.boxIndex());
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::updateTopology()
	{
		this->requestDiscreteElementsInGlobal(mBoxInGlobal, mSphereInGlobal, mTetInGlobal, mCapsuleInGlobal, mTriangleInGlobal);
	}

	DEFINE_CLASS(DiscreteElements);
}