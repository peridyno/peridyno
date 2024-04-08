#pragma once

#include "Vector.h"
#include "Matrix.h"
#include "Quat.h"


namespace dyno
{
	template<typename Real>
	class Joint
	{
	public:
		DYN_FUNC Joint()
		{
			this->bodyId1 = INVALID;
			this->bodyId2 = INVALID;
		}
		DYN_FUNC Joint(int bodyId1, int bodyId2)
		{
			this->bodyId1 = bodyId1;
			this->bodyId2 = bodyId2;
		}
	public:
		int bodyId1;
		int bodyId2;
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
		
		void setAnchorPoint(Vector<Real, 3>anchor_point, Vector<Real, 3> body1_pos, Vector<Real, 3>body2_pos)
		{
			this->r1 = anchor_point - body1_pos;
			this->r2 = anchor_point - body2_pos;
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

		void setAnchorPoint(Vector<Real, 3>anchor_point, Vector<Real, 3> body1_pos, Vector<Real, 3>body2_pos)
		{
			this->r1 = anchor_point - body1_pos;
			this->r2 = anchor_point - body2_pos;
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

		void setAnchorPoint(Vector<Real, 3>anchor_point, Vector<Real, 3> body1_pos, Vector<Real, 3>body2_pos)
		{
			this->r1 = anchor_point - body1_pos;
			this->r2 = anchor_point - body2_pos;
		}
	public:
		// anchor point position in body1 and body2 local space
		Vector<Real, 3> r1;
		Vector<Real, 3> r2;
	};
}