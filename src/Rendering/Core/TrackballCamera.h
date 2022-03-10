#pragma once
#include "Camera.h"

namespace dyno
{
	class TrackballCamera : public Camera {

	public:
		TrackballCamera();
		~TrackballCamera() {};

		void reset();

		void registerPoint(float x, float y) override;

		void rotateToPoint(float x, float y) override;
		void translateToPoint(float x, float y) override;
		void zoom(float amount) override;

		//TODO: implement
		void setEyePos(const Vec3f& p) override {};

		//TODO: implement
		void setTargetPos(const Vec3f& p) override {};

		glm::mat4 getViewMat() override;
		glm::mat4 getProjMat() override;

	public:
		float mRegX;
		float mRegY;

		Vec3f mCameraPos;
		Vec3f mCameraTarget;
		Vec3f mCameraUp;
	};

}

