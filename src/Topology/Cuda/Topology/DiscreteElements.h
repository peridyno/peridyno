#pragma once
#include "Module/TopologyModule.h"
#include "Primitive/Primitive3D.h"

namespace dyno
{
	enum ElementType
	{
		ET_BOX = 1,
		ET_TET = 2,
		ET_CAPSULE = 4,
		ET_SPHERE = 8,
		ET_TRI = 16,
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
		}

		DYN_FUNC Joint(int bodyId1, int bodyId2)
		{
			this->bodyId1 = bodyId1;
			this->bodyId2 = bodyId2;

			this->bodyType1 = ET_Other;
			this->bodyType2 = ET_Other;
		}
	public:
		int bodyId1;
		int bodyId2;

		ElementType bodyType1;
		ElementType bodyType2;
	};


	template<typename Real>
	class BallAndSocketJoint : public Joint<Real>
	{
	public:
		DYN_FUNC BallAndSocketJoint()
		{
			this->bodyId1 = INVALID;
			this->bodyId2 = INVALID;
		}
		DYN_FUNC BallAndSocketJoint(int bodyId1, int bodyId2)
		{
			this->bodyId1 = bodyId1;
			this->bodyId2 = bodyId2;
		}

		void setAnchorPoint(Vector<Real, 3>anchor_point, Vector<Real, 3> body1_pos, Vector<Real, 3>body2_pos, Quat<Real> body1_quat, Quat<Real> body2_quat)
		{
			Mat3f rotMat1 = body1_quat.toMatrix3x3();
			Mat3f rotMat2 = body2_quat.toMatrix3x3();
			this->r1 = rotMat1.inverse() * (anchor_point - body1_pos);
			this->r2 = rotMat2.inverse() * (anchor_point - body2_pos);
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
		}

		DYN_FUNC SliderJoint(int bodyId1, int bodyId2)
		{
			this->bodyId1 = bodyId1;
			this->bodyId2 = bodyId2;
		}

		void setAnchorPoint(Vector<Real, 3>anchor_point, Vector<Real, 3> body1_pos, Vector<Real, 3>body2_pos, Quat<Real> body1_quat, Quat<Real> body2_quat)
		{
			Mat3f rotMat1 = body1_quat.toMatrix3x3();
			Mat3f rotMat2 = body2_quat.toMatrix3x3();
			this->r1 = rotMat1.inverse() * (anchor_point - body1_pos);
			this->r2 = rotMat2.inverse() * (anchor_point - body2_pos);
		}

		void setAxis(Vector<Real, 3> axis)
		{
			this->sliderAxis = axis;
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
	};


	template<typename Real>
	class HingeJoint : public Joint<Real>
	{
	public:
		DYN_FUNC HingeJoint()
		{
			this->bodyId1 = INVALID;
			this->bodyId2 = INVALID;
		}
		DYN_FUNC HingeJoint(int bodyId1, int bodyId2)
		{
			this->bodyId1 = bodyId1;
			this->bodyId2 = bodyId2;
		}

		void setAnchorPoint(Vector<Real, 3>anchor_point, Vector<Real, 3> body1_pos, Vector<Real, 3>body2_pos, Quat<Real> body1_quat, Quat<Real> body2_quat)
		{
			Mat3f rotMat1 = body1_quat.toMatrix3x3();
			Mat3f rotMat2 = body2_quat.toMatrix3x3();
			this->r1 = rotMat1.inverse() * (anchor_point - body1_pos);
			this->r2 = rotMat2.inverse() * (anchor_point - body2_pos);
		}

		void setAxis(Vector<Real, 3> axis, Quat<Real> quat_1, Quat<Real> quat_2)
		{
			Mat3f rotMat1 = quat_1.toMatrix3x3();
			Mat3f rotMat2 = quat_2.toMatrix3x3();
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
		}

		DYN_FUNC FixedJoint(int bodyId1, int bodyId2)
		{
			this->bodyId1 = bodyId1;
			this->bodyId2 = bodyId2;
		}

		void setAnchorPoint(Vector<Real, 3>anchor_point, Vector<Real, 3> body1_pos, Vector<Real, 3>body2_pos, Quat<Real> body1_quat, Quat<Real> body2_quat)
		{
			Mat3f rotMat1 = body1_quat.toMatrix3x3();
			Mat3f rotMat2 = body2_quat.toMatrix3x3();
			this->r1 = rotMat1.inverse() * (anchor_point - body1_pos);
			this->r2 = rotMat2.inverse() * (anchor_point - body2_pos);
		}
	public:
		// anchor point position in body1 and body2 local space
		Vector<Real, 3> r1;
		Vector<Real, 3> r2;
	};


	template<typename Real>
	class PointJoint : public Joint<Real>
	{
	public:
		PointJoint()
		{
			this->bodyId1 = INVALID;
			this->bodyId2 = INVALID;
		}
		PointJoint(int bodyId)
		{
			this->bodyId1 = bodyId;
			this->bodyId2 = INVALID;
		}
		void setAnchorPoint(Vector<Real, 3> point)
		{
			this->anchorPoint = point;
		}

	public:
		Vector<Real, 3> anchorPoint;

	};


	/**
	 * Discrete elements will arranged in the order of sphere, box, tet, capsule, triangle
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

		typedef typename BallAndSocketJoint<Real> BallAndSocketJoint;
		typedef typename SliderJoint<Real> SliderJoint;
		typedef typename HingeJoint<Real> HingeJoint;
		typedef typename FixedJoint<Real> FixedJoint;
		typedef typename PointJoint<Real> PointJoint;

		DiscreteElements();
		~DiscreteElements() override;

		void scale(Real s);

		uint totalSize();

		uint totalJointSize();

		uint sphereIndex();
		uint boxIndex();
		uint capsuleIndex();
		uint tetIndex();
		uint triangleIndex();

		ElementOffset calculateElementOffset();

		void setBoxes(DArray<Box3D>& boxes);
		void setSpheres(DArray<Sphere3D>& spheres);
		void setTets(DArray<Tet3D>& tets);
		void setCapsules(DArray<Capsule3D>& capsules);
		void setTriangles(DArray<Triangle3D>& triangles);
		void setTetSDF(DArray<Real>& sdf);

		DArray<Box3D>&		getBoxes() { return m_boxes; }
		DArray<Sphere3D>&	getSpheres() { return m_spheres; }
		DArray<Tet3D>&		getTets() { return m_tets; }
		DArray<Capsule3D>&	getCaps() { return m_caps; }
		DArray<Triangle3D>& getTris() { return m_tris; }

		DArray<BallAndSocketJoint>& ballAndSocketJoints() { return mBallAndSocketJoints; };
		DArray<SliderJoint>& sliderJoints() { return mSliderJoints; };
		DArray<HingeJoint>& hingeJoints() { return mHingeJoints; };
		DArray<FixedJoint>& fixedJoints() { return mFixedJoints; };
		DArray<PointJoint>& pointJoints() { return mPointJoints; };

		void setTetBodyId(DArray<int>& body_id);
		void setTetElementId(DArray<TopologyModule::Tetrahedron>& element_id);

		DArray<Real>&		getTetSDF() { return m_tet_sdf; }
		DArray<int>&		getTetBodyMapping() { return m_tet_body_mapping; }
		DArray<TopologyModule::Tetrahedron>& getTetElementMapping() { return m_tet_element_id; }

	protected:
		DArray<Sphere3D> m_spheres;
		DArray<Box3D> m_boxes;
		DArray<Tet3D> m_tets;
		DArray<Capsule3D> m_caps;
		DArray<Triangle3D> m_tris;

		DArray<BallAndSocketJoint> mBallAndSocketJoints;
		DArray<SliderJoint> mSliderJoints;
		DArray<HingeJoint> mHingeJoints;
		DArray<FixedJoint> mFixedJoints;
		DArray<PointJoint> mPointJoints;
		
		DArray<Real> m_tet_sdf;
		DArray<int> m_tet_body_mapping;
		DArray<TopologyModule::Tetrahedron> m_tet_element_id;
	};
}

