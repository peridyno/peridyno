#pragma once
#include <vector>
#include "Quat.h"
#include "Vector.h"

#include <glm/mat4x4.hpp>

namespace dyno
{
	typedef Quat<float> Quat1f;

	class Camera {

	public:
		Camera();
		~Camera() {};

		void reset();

		void rotateToPoint(float x, float y);
		void translateToPoint(float x, float y);
		void zoom(float amount);

		void setWidth(int width) { mViewportWidth = width; }
		void setHeight(int height) { mViewportHeight = height; }

		void setClipNear(float zNear) { mNear = zNear; }
		void setClipFar(float zFar) { mFar = zFar; }

		glm::mat4 getViewMat();
		glm::mat4 getProjMat();

		
		void registerPoint(float x, float y);

	private:
		void rotate(float dx, float dy);
		void translate(const Vec3f translation);

		Vec3f getPosition(float x, float y);
		Quat1f getQuaternion(float x1, float y1, float x2, float y2);

	public:
		float mRegX;
		float mRegY;

		float mNear;
		float mFar;
		float mFov;

		float mRotAngle;
		float mFocusDist;
		
		float mYaw;	//along Y
		float mPitch;	//along axis X of the viewport coordinate system

		int mViewportWidth;
		int mViewportHeight;

		Vec3f mEyePos;
		Vec3f mTargetPos;
		Vec3f mRotAxis;
	};

}

