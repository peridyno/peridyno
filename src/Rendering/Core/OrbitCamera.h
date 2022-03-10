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
		Vec3f getTargetPos() const;

		void setEyePos(const Vec3f& p) override;
		void setTargetPos(const Vec3f& p) override;

		void getCoordSystem(Vec3f &view, Vec3f &up, Vec3f &right) const;

		glm::mat4 getViewMat() override;
		glm::mat4 getProjMat() override;

	private:
		void rotate(float dx, float dy);
		void translate(const Vec3f translation);

		Vec3f getPosition(float x, float y);
		Quat1f getQuaternion(float x1, float y1, float x2, float y2);

		Quat1f getQuaternion(float yaw, float pitch) const;

	private:
		float mRegX = 0.5f;
		float mRegY = 0.5f;

		//Auxiliary parameters to form a right-hand coordinate or left-hand side coordinate
		float mRotAngle = 0.0f;
		Vec3f mRotAxis = Vec3f(0.0f, 1.0f, 0.0f);

		float mYaw = 0.0f;		//along Y
		float mPitch = 0.0f;	//along axis X of the viewport coordinate system

		float mFocusDist;

		Vec3f mEyePos;
		Vec3f mTargetPos;
		
		float mFocusDistMax = 10.0f;
		float mFocusDistMin = 0.1f;

		float mSpeed = 2.0;
		float mZoomSpeed = 1.0f;
	};

}

