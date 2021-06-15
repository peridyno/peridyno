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


	public:
		float mRegX;
		float mRegY;

		float mNear;
		float mFar;
		float mFov;

		int mViewportWidth;
		int mViewportHeight;

		Vec3f mCameraPos;
		Vec3f mCameraTarget;
		Vec3f mCameraUp;

	};

}

