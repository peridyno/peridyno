#pragma once
#include "Module/TopologyModule.h"
#include "Primitive/Primitive3D.h"

#include "STL/Pair.h"

namespace dyno
{
	enum ElementType
	{
		ET_BOX = 1,
		ET_TET = 2,
		ET_CAPSULE = 4,
		ET_SPHERE = 8,
		ET_TRI = 16,
		ET_COMPOUND = 32,
		ET_MEDIALCONE = 64,
		ET_MEDIALSLAB = 128,
		ET_Other = 0x80000000
	};

	struct ElementOffset
	{
	public:
		DYN_FUNC inline uint sphereIndex() { return sphereStart; }
		DYN_FUNC inline uint boxIndex() { return boxStart; }
		DYN_FUNC inline uint tetIndex() { return tetStart; }
		DYN_FUNC inline uint capsuleIndex() { return capStart; }
		DYN_FUNC inline uint triangleIndex() { return triStart; }
		DYN_FUNC inline uint medialConeIndex() { return medialConeStart; }
		DYN_FUNC inline uint medialSlabIndex() { return medialSlabStart; }

		DYN_FUNC inline void setSphereRange(uint startIndex, uint endIndex) { 
			sphereStart = startIndex;
			sphereEnd = endIndex;
		}

		DYN_FUNC inline void setBoxRange(uint startIndex, uint endIndex) {
			boxStart = startIndex;
			boxEnd = endIndex;
		}

		DYN_FUNC inline void setTetRange(uint startIndex, uint endIndex) {
			tetStart = startIndex;
			tetEnd = endIndex;
		}

		DYN_FUNC inline void setCapsuleRange(uint startIndex, uint endIndex) {
			capStart = startIndex;
			capEnd = endIndex;
		}

		DYN_FUNC inline void setTriangleRange(uint startIndex, uint endIndex) {
			triStart = startIndex;
			triEnd = endIndex;
		}

		DYN_FUNC inline void setMedialConeRange(uint startIndex, uint endIndex) {
			medialConeStart = startIndex;
			medialConeEnd = endIndex;
		}

		DYN_FUNC inline void setMedialSlabRange(uint startIndex, uint endIndex) {
			medialSlabStart = startIndex;
			medialSlabEnd = endIndex;
		}

		DYN_FUNC inline uint checkElementOffset(ElementType eleType)
		{
			if (eleType == ET_SPHERE)
				return sphereStart;

			if (eleType == ET_BOX)
				return boxStart;

			if (eleType == ET_TET)
				return tetStart;

			if (eleType == ET_CAPSULE)
				return capStart;

			if (eleType == ET_TRI)
				return triStart;

			if (eleType == ET_MEDIALCONE)
				return medialConeStart;

			if (eleType == ET_MEDIALSLAB)
				return medialSlabStart;

			return 0;
		}

		DYN_FUNC inline ElementType checkElementType(uint id)
		{
			if (id >= sphereStart && id < sphereEnd)
				return ET_SPHERE;

			if (id >= boxStart && id < boxEnd)
				return ET_BOX;

			if (id >= tetStart && id < tetEnd)
				return ET_TET;

			if (id >= capStart && id < capEnd)
				return ET_CAPSULE;

			if (id >= triStart && id < triEnd)
				return ET_TRI;

			if (id >= medialConeStart && id < medialConeEnd)
				return ET_MEDIALCONE;

			if (id >= medialSlabStart && id < medialSlabEnd)
				return ET_MEDIALSLAB;
		}

	private:
		uint sphereStart;
		uint sphereEnd;
		uint boxStart;
		uint boxEnd;
		uint tetStart;
		uint tetEnd;
		uint capStart;
		uint capEnd;
		uint triStart;
		uint triEnd;
		uint medialConeStart;
		uint medialConeEnd;
		uint medialSlabStart;
		uint medialSlabEnd;
	};

	class PdActor
	{
	public:
		int idx = INVALID;

		ElementType shapeType = ET_Other;

		

		Vec3f center;

		Quat1f rot;
	};

	template<typename Real>
	class Joint
	{
	public:
		DYN_FUNC Joint()
		{
			this->bodyId1 = INVALID;
			this->bodyId2 = INVALID;

			this->bodyType1 = ET_Other;
			this->bodyType2 = ET_Other;

			this->actor1 = nullptr;
			this->actor2 = nullptr;
		}

		CPU_FUNC Joint(PdActor* a1, PdActor* a2)
		{
			this->bodyId1 = a1->idx;
			this->bodyId2 = a2->idx;

			this->bodyType1 = a1->shapeType;
			this->bodyType2 = a2->shapeType;

			this->actor1 = a1;
			this->actor2 = a2;
		}

	public:
		int bodyId1;
		int bodyId2;

		ElementType bodyType1;
		ElementType bodyType2;

		//The following two pointers should only be visited from host codes.
		PdActor* actor1 = nullptr;
		PdActor* actor2 = nullptr;
	};


	template<typename Real>
	class BallAndSocketJoint : public Joint<Real>
	{
	public:
		DYN_FUNC BallAndSocketJoint()
		{
			this->bodyId1 = INVALID;
			this->bodyId2 = INVALID;

			this->bodyType1 = ET_Other;
			this->bodyType2 = ET_Other;

			this->actor1 = nullptr;
			this->actor2 = nullptr;
		}

		CPU_FUNC BallAndSocketJoint(PdActor* a1, PdActor* a2)
		{
			this->bodyId1 = a1->idx;
			this->bodyId2 = a2->idx;

			this->bodyType1 = a1->shapeType;
			this->bodyType2 = a2->shapeType;

			this->actor1 = a1;
			this->actor2 = a2;
		}

		void setAnchorPoint(Vector<Real, 3>anchor_point)
		{
			Mat3f rotMat1 = this->actor1->rot.toMatrix3x3();
			Mat3f rotMat2 = this->actor2->rot.toMatrix3x3();
			this->r1 = rotMat1.inverse() * (anchor_point - this->actor1->center);
			this->r2 = rotMat2.inverse() * (anchor_point - this->actor2->center);
		}

	public:
		// anchor point in body1 local space
		Vector<Real, 3> r1;
		// anchor point in body2 local space
		Vector<Real, 3> r2;
	};

	template<typename Real>
	class SliderJoint : public Joint<Real>
	{
	public:
		DYN_FUNC SliderJoint()
		{
			this->bodyId1 = INVALID;
			this->bodyId2 = INVALID;

			this->bodyType1 = ET_Other;
			this->bodyType2 = ET_Other;

			this->actor1 = nullptr;
			this->actor2 = nullptr;
		}

		CPU_FUNC SliderJoint(PdActor* a1, PdActor* a2)
		{
			this->bodyId1 = a1->idx;
			this->bodyId2 = a2->idx;

			this->bodyType1 = a1->shapeType;
			this->bodyType2 = a2->shapeType;

			this->actor1 = a1;
			this->actor2 = a2;
		}

		void setAnchorPoint(Vector<Real, 3>anchor_point)
		{
			Mat3f rotMat1 = this->actor1->rot.toMatrix3x3();
			this->r1 = rotMat1.inverse() * (anchor_point - this->actor1->center);
			if (this->bodyId2 != INVALID)
			{
				Mat3f rotMat2 = this->actor2->rot.toMatrix3x3();
				this->r2 = rotMat2.inverse() * (anchor_point - this->actor2->center);
				this->q_init = this->actor2->rot.inverse() * this->actor1->rot;
			}
			else
			{
				this->q_init = this->actor1->rot;
			}
		}

		void setAxis(Vector<Real, 3> axis)
		{
			Mat3f rotMat1 = this->actor1->rot.toMatrix3x3();
			this->sliderAxis = rotMat1.transpose() * axis;
		}

		void setMoter(Real v_moter)
		{
			this->useMoter = true;
			this->v_moter = v_moter;
		}

		void setRange(Real d_min, Real d_max)
		{
			this->d_min = d_min;
			this->d_max = d_max;
			this->useRange = true;
		}


	public:
		bool useRange = false;
		bool useMoter = false;
		// motion range
		Real d_min;
		Real d_max;
		Real v_moter;
		// anchor point position in body1 and body2 local space
		Vector<Real, 3> r1;
		Vector<Real, 3> r2;
		// slider axis in body1 local space
		Vector<Real, 3> sliderAxis;

		Quat1f q_init;
	};


	template<typename Real>
	class HingeJoint : public Joint<Real>
	{
	public:
		DYN_FUNC HingeJoint()
		{
			this->bodyId1 = INVALID;
			this->bodyId2 = INVALID;

			this->bodyType1 = ET_Other;
			this->bodyType2 = ET_Other;

			this->actor1 = nullptr;
			this->actor2 = nullptr;
		}

		CPU_FUNC HingeJoint(PdActor* a1, PdActor* a2)
		{
			this->bodyId1 = a1->idx;
			this->bodyId2 = a2->idx;

			this->bodyType1 = a1->shapeType;
			this->bodyType2 = a2->shapeType;

			this->actor1 = a1;
			this->actor2 = a2;
		}

		void setAnchorPoint(Vector<Real, 3>anchor_point)
		{
			Mat3f rotMat1 = this->actor1->rot.toMatrix3x3();
			Mat3f rotMat2 = this->actor2->rot.toMatrix3x3();
			this->r1 = rotMat1.inverse() * (anchor_point - this->actor1->center);
			this->r2 = rotMat2.inverse() * (anchor_point - this->actor2->center);
		}

		void setAxis(Vector<Real, 3> axis)
		{
			Mat3f rotMat1 = this->actor1->rot.toMatrix3x3();
			Mat3f rotMat2 = this->actor2->rot.toMatrix3x3();
			this->hingeAxisBody1 = rotMat1.inverse() * axis;
			this->hingeAxisBody2 = rotMat2.inverse() * axis;
		}

		void setRange(Real theta_min, Real theta_max)
		{
			this->d_min = theta_min;
			this->d_max = theta_max;
			this->useRange = true;
		}

		void setMoter(Real v_moter)
		{
			this->v_moter = v_moter;
			this->useMoter = true;
		}

	public:
		// motion range
		Real d_min;
		Real d_max;
		Real v_moter;
		// anchor point position in body1 and body2 local space
		Vector<Real, 3> r1;
		Vector<Real, 3> r2;

		// axis a in body local space
		Vector<Real, 3> hingeAxisBody1;
		Vector<Real, 3> hingeAxisBody2;

		bool useMoter = false;
		bool useRange = false;
	};

	template<typename Real>
	class FixedJoint : public Joint<Real>
	{
	public:
		DYN_FUNC FixedJoint()
		{
			this->bodyId1 = INVALID;
			this->bodyId2 = INVALID;

			this->bodyType1 = ET_Other;
			this->bodyType2 = ET_Other;

			this->actor1 = nullptr;
			this->actor2 = nullptr;
		}

		CPU_FUNC FixedJoint(PdActor* a1, PdActor* a2)
		{
			this->bodyId1 = a1->idx;
			this->bodyId2 = a2->idx;

			this->bodyType1 = a1->shapeType;
			this->bodyType2 = a2->shapeType;

			this->actor1 = a1;
			this->actor2 = a2;
		}

		CPU_FUNC FixedJoint(PdActor* a1)
		{
			this->bodyId1 = a1->idx;
			this->bodyId2 = INVALID;

			this->bodyType1 = a1->shapeType;
			this->bodyType2 = ET_Other;

			this->actor1 = a1;
			this->actor2 = nullptr;
		}

		void setAnchorPoint(Vector<Real, 3>anchor_point)
		{
			Mat3f rotMat1 = this->actor1->rot.toMatrix3x3();
			this->r1 = rotMat1.inverse() * (anchor_point - this->actor1->center);
			this->w = anchor_point;
			if (this->bodyId2 != INVALID)
			{
				Mat3f rotMat2 = this->actor2->rot.toMatrix3x3();
				this->r2 = rotMat2.inverse() * (anchor_point - this->actor2->center);
				this->q_init = this->actor2->rot.inverse() * this->actor1->rot;
			}
			else
			{
				this->q_init = this->actor1->rot;
			}
		}

		void setAnchorAngle(Quat<Real> quat) { q = quat; }

	public:
		// anchor point position in body1 and body2 local space
		Vector<Real, 3> r1;
		Vector<Real, 3> r2;
		Vector<Real, 3> w;
		Quat<Real> q;
		Quat<Real> q_init;
	};



	template<typename Real>
	class PointJoint : public Joint<Real>
	{
	public:
		PointJoint()
		{
			this->bodyId1 = INVALID;
			this->bodyId2 = INVALID;

			this->bodyType1 = ET_Other;
			this->bodyType2 = ET_Other;

			this->actor1 = nullptr;
			this->actor2 = nullptr;
		}
		PointJoint(PdActor* a1)
		{
			this->bodyId1 = a1->idx;
			this->bodyId2 = INVALID;

			this->bodyType1 = a1->shapeType;
			this->bodyType2 = ET_Other;

			this->actor1 = a1;
			this->actor2 = nullptr;
		}
		void setAnchorPoint(Vector<Real, 3> point)
		{
			this->anchorPoint = point;
		}

	public:
		Vector<Real, 3> anchorPoint;

	};

	template<typename Real>
	class DistanceJoint : public Joint<Real>
	{
	public:
		DistanceJoint()
		{
			this->bodyId1 = INVALID;
			this->bodyId2 = INVALID;

			this->bodyType1 = ET_Other;
			this->bodyType2 = ET_Other;

			this->actor1 = nullptr;
			this->actor2 = nullptr;
		}
		DistanceJoint(PdActor* a1, PdActor* a2)
		{
			this->bodyId1 = a1->idx;
			this->bodyId2 = a2->idx;

			this->bodyType1 = a1->shapeType;
			this->bodyType2 = a2->shapeType;

			this->actor1 = a1;
			this->actor2 = a2;
		}
		void setDistanceJoint(Vector<Real, 3> r1, Vector<Real, 3> r2, Real distance)
		{
			this->r1 = r1;
			this->r2 = r2;
			this->distance = distance;
		}
	public:
		// anchor point position in body1 and body2 local space
		Vector<Real, 3> r1;
		Vector<Real, 3> r2;
		Real distance;
	};

	/**
	 * Discrete elements will arranged in the order of sphere, box, tet, capsule, triangle, medialCone, medialSlab
	 */
	template<typename TDataType>
	class DiscreteElements : public TopologyModule
	{
		DECLARE_TCLASS(DiscreteElements, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename ::dyno::TSphere3D<Real> Sphere3D;
		typedef typename ::dyno::TOrientedBox3D<Real> Box3D;
		typedef typename ::dyno::TTet3D<Real> Tet3D;

		typedef typename ::dyno::BallAndSocketJoint<Real> BallAndSocketJoint;
		typedef typename ::dyno::SliderJoint<Real> SliderJoint;
		typedef typename ::dyno::HingeJoint<Real> HingeJoint;
		typedef typename ::dyno::FixedJoint<Real> FixedJoint;
		typedef typename ::dyno::PointJoint<Real> PointJoint;
		typedef typename ::dyno::DistanceJoint<Real> DistanceJoint;

		DiscreteElements();
		~DiscreteElements() override;

		void scale(Real s);

		uint totalSize();

		uint totalJointSize();

		uint sphereIndex();
		uint boxIndex();
		uint tetIndex();
		uint capsuleIndex();
		uint triangleIndex();
		uint medialConeIndex();
		uint medialSlabIndex();

		ElementOffset calculateElementOffset();

		//Set basic shapes in local frame
		void setSpheres(DArray<Sphere3D>& spheres);
		void setBoxes(DArray<Box3D>& boxes);
		void setTets(DArray<Tet3D>& tets);
		void setCapsules(DArray<Capsule3D>& capsules);
		void setTriangles(DArray<Triangle3D>& triangles);
		void setTetSDF(DArray<Real>& sdf);
		void setMedialCones(DArray<MedialCone3D>& medialcones);
		void setMedialSlabs(DArray<MedialSlab3D>& medialsalbs);

		DArray<Sphere3D>&	spheresInLocal() { return mSpheresInLocal; }
		DArray<Box3D>&		boxesInLocal() { return mBoxesInLocal; }
		DArray<Tet3D>&		tetsInLocal() { return mTetsInLocal; }
		DArray<Capsule3D>&	capsulesInLocal() { return mCapsulesInLocal; }
		DArray<Triangle3D>&	trianglesInLocal() { return mTrianglesInLocal; }
		DArray<MedialCone3D>& medialConesInLocal() { return mMedialConesInLocal; }
		DArray<MedialSlab3D>& medialSlabsInLocal() { return mMedialSlabsInLocal; }

		DArray<Sphere3D>&	spheresInGlobal() { return mSphereInGlobal; }
		DArray<Box3D>&		boxesInGlobal() { return mBoxInGlobal; }
		DArray<Tet3D>&		tetsInGlobal() { return mTetInGlobal; }
		DArray<Capsule3D>&	capsulesInGlobal() { return mCapsuleInGlobal; }
		DArray<Triangle3D>& trianglesInGlobal() { return mTriangleInGlobal; }
		DArray<MedialCone3D>& medialConesInGlobal() { return mMedialConesInGlobal; }
		DArray<MedialSlab3D>& medialSlabsInGlobal() { return mMedialSlabsInGlobal; }

		DArray<Pair<uint, uint>>& shape2RigidBodyMapping() { return mShape2RigidBody; };

		DArray<Coord>& position() { return mPosition; }
		DArray<Matrix>& rotation() { return mRotation; }

		void setPosition(const DArray<Coord>& pos) { mPosition.assign(pos); }
		void setRotation(const DArray<Matrix>& rot) { mRotation.assign(rot); }

		DArray<BallAndSocketJoint>& ballAndSocketJoints() { return mBallAndSocketJoints; };
		DArray<SliderJoint>& sliderJoints() { return mSliderJoints; };
		DArray<HingeJoint>& hingeJoints() { return mHingeJoints; };
		DArray<FixedJoint>& fixedJoints() { return mFixedJoints; };
		DArray<PointJoint>& pointJoints() { return mPointJoints; };
		DArray<DistanceJoint>& distanceJoints() { return mDistanceJoints; };

		void setTetBodyId(DArray<int>& body_id);
		void setTetElementId(DArray<TopologyModule::Tetrahedron>& element_id);

		DArray<Real>&		getTetSDF() { return m_tet_sdf; }
		DArray<int>&		getTetBodyMapping() { return m_tet_body_mapping; }
		DArray<TopologyModule::Tetrahedron>& getTetElementMapping() { return m_tet_element_id; }

		void copyFrom(DiscreteElements<TDataType>& de);

		void merge(CArray<std::shared_ptr<DiscreteElements<TDataType>>>& topos);

		void requestDiscreteElementsInGlobal(
			DArray<Box3D>& boxInGlobal,
			DArray<Sphere3D>& sphereInGlobal,
			DArray<Tet3D>& tetInGlobal,
			DArray<Capsule3D>& capInGlobal,
			DArray<Triangle3D>& triInGlobal,
			DArray<MedialCone3D>& medialConeInGlobal,
			DArray<MedialSlab3D>& medialSlabInGlobal);

		void requestBoxInGlobal(DArray<Box3D>& boxInGlobal);
		void requestSphereInGlobal(DArray<Sphere3D>& sphereInGlobal);
		void requestTetInGlobal(DArray<Tet3D>& tetInGlobal);
		void requestCapsuleInGlobal(DArray<Capsule3D>& capInGlobal);
		void requestTriangleInGlobal(DArray<Triangle3D>& triInGlobal);

	protected:
		void updateTopology() override;

	protected:
		DArray<Sphere3D> mSpheresInLocal;
		DArray<Box3D> mBoxesInLocal;
		DArray<Tet3D> mTetsInLocal;
		DArray<Capsule3D> mCapsulesInLocal;
		DArray<Triangle3D> mTrianglesInLocal;
		DArray<MedialCone3D> mMedialConesInLocal;
		DArray<MedialSlab3D> mMedialSlabsInLocal;

		DArray<Sphere3D> mSphereInGlobal;
		DArray<Box3D> mBoxInGlobal;
		DArray<Tet3D> mTetInGlobal;
		DArray<Capsule3D> mCapsuleInGlobal;
		DArray<Triangle3D> mTriangleInGlobal;
		DArray<MedialCone3D> mMedialConesInGlobal;
		DArray<MedialSlab3D> mMedialSlabsInGlobal;

		DArray<BallAndSocketJoint> mBallAndSocketJoints;
		DArray<SliderJoint> mSliderJoints;
		DArray<HingeJoint> mHingeJoints;
		DArray<FixedJoint> mFixedJoints;
		DArray<PointJoint> mPointJoints;
		DArray<DistanceJoint> mDistanceJoints;

		DArray<Pair<uint, uint>> mShape2RigidBody;

		DArray<Coord> mPosition;
		DArray<Matrix> mRotation;

		DArray<Real> m_tet_sdf;
		DArray<int> m_tet_body_mapping;
		DArray<TopologyModule::Tetrahedron> m_tet_element_id;
	};
}

