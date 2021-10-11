#pragma once

#include "Camera.h"

namespace dyno
{
	class OrbitCamera : public Camera {

	public:
		OrbitCamera();
		~OrbitCamera() {};

		void rotateToPoint(float x, float y) override;
		void translateToPoint(float x, float y) override;
		void zoom(float amount) override;

		void registerPoint(float x, float y) override;

		Vec3f getViewDir() const;
		Vec3f getEyePos() const;

		void getCoordSystem(Vec3f &view, Vec3f &up, Vec3f &right) const;

		glm::mat4 getViewMat() override;
		glm::mat4 getProjMat() override;

	private:
		void rotate(float dx, float dy);
		void translate(const Vec3f translation);

		Vec3f getPosition(float x, float y);
		Quat1f getQuaternion(float x1, float y1, float x2, float y2);

	public:
		float mRegX;
		float mRegY;

		float mRotAngle;
		float mFocusDist;
		
		float mYaw;	//along Y
		float mPitch;	//along axis X of the viewport coordinate system

		Vec3f mEyePos;
		Vec3f mTargetPos;
		Vec3f mRotAxis;

		float mFocusDistMax = 10.0f;
		float mFocusDistMin = 0.1f;

		float mSpeed = 2.0;
		float mZoomSpeed = 1.0f;
	};

}

