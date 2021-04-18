#pragma once
#include <vector>
#include "Quat.h"
#include "Vector.h"

namespace dyno
{
	typedef Quat<float> Quat1f;

	class Camera {

	public:
		Camera();
		~Camera() {};

		void rotateToPoint(float x, float y);
		void translateToPoint(float x, float y);
		void translateLightToPoint(float x, float y);
		void zoom(float amount);
		void setGL(float neardist, float fardist, float width, float height);

		int viewportWidth() const;
		int viewportHeight() const;

		Vec3f getViewDir() const;
		Vec3f getEyePos() const;

		void getCoordSystem(Vec3f &view, Vec3f &up, Vec3f &right) const;

		void registerPoint(float x, float y);

	private:
		void rotate(float dx, float dy);
		void translate(const Vec3f translation);
		void translateLight(const Vec3f translation);

		Vec3f getPosition(float x, float y);

		Quat1f getQuaternion(float x1, float y1, float x2, float y2);

	private:
		float mRegX;
		float mRegY;

		float mNear;
		float mFar;
		float mRight;
		float mFov;

		float mRotAngle;
		float mFocusDist;
		
		float mYaw;	//along Y
		float mPitch;	//along axis X of the viewport coordinate system

		int mViewportWidth;
		int mViewportHeight;

		Vec3f mEyePos;
		Vec3f mTargetPos;
		Vec3f mLightPos;
		Vec3f mRotAxis;
	};

}

