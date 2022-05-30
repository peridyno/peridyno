#include "TrackballCamera.h"

#include <iostream>
#include <math.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include <Vector.h>

namespace dyno
{
	TrackballCamera::TrackballCamera() 
		: Camera()
	{
		mCameraPos = Vec3f(0.0f, 0.0f, 3.0f);
		mCameraTarget = Vec3f(0, 0, 0);
		mCameraUp = Vec3f(0, 1, 0);

		mFov = glm::radians(45.0f);
	}

	void TrackballCamera::reset()
	{
		// TODO: reset to fit scene
	}
	
	glm::mat4 TrackballCamera::getViewMat()
	{
		return glm::lookAt(mCameraPos.data_, mCameraTarget.data_, mCameraUp.data_);
	}

	glm::mat4 TrackballCamera::getProjMat()
	{
		float aspect = std::max(float(mViewportWidth), 1.0f) / std::max(float(mViewportHeight), 1.0f);

		glm::mat4 projection;

		if (mProjectionType == Perspective)
		{
			projection = glm::perspective(mFov, aspect, mNear * mDistanceUnit, mFar * mDistanceUnit);
		}
		else
		{
			float half_depth = (mCameraPos - mCameraTarget).norm() * mDistanceUnit;
			projection = glm::ortho(-half_depth * aspect, half_depth * aspect, -half_depth, half_depth, -5.0f * half_depth, 5.0f * half_depth);
		}
		
		return projection;
	}

	void TrackballCamera::zoom(float amount) 
	{
		mFov += amount / 10;
		mFov = std::max(mFov, 0.05f);

		// maybe we want to move forward/backward the camera...
		Vec3f viewDir = mCameraPos - mCameraTarget;
		Vec3f t = viewDir * (amount / 10.0);
		mCameraPos += t;
		mCameraTarget += t;
	}
		
	void TrackballCamera::registerPoint(float xpos, float ypos) {
		mRegX = float(xpos) / float(mViewportWidth) - 0.5f;
		mRegY = float(mViewportHeight - ypos) / float(mViewportHeight) - 0.5f;
	}

	void TrackballCamera::rotateToPoint(float xpos, float ypos) {
		float x = float(xpos) / float(mViewportWidth) - 0.5f;
		float y = float(mViewportHeight - ypos) / float(mViewportHeight) - 0.5f;

		float dx = x - mRegX;
		float dy = y - mRegY;

		Vec3f viewDir = mCameraPos - mCameraTarget;
		Vec3f rightDir = mCameraUp.cross(viewDir).normalize();
		Vec3f upDir = viewDir.cross(rightDir).normalize();

		viewDir.data_ = glm::rotate(viewDir.data_, dy, rightDir.data_);
		viewDir.data_ = glm::rotate(viewDir.data_, -dx, upDir.data_);
		mCameraPos = mCameraTarget + viewDir;
		
		mCameraUp.data_ = glm::rotate(upDir.data_, dy, rightDir.data_);
		mCameraUp.data_ = glm::rotate(upDir.data_, -dx, upDir.data_);

		registerPoint(xpos, ypos);
	}

	void TrackballCamera::translateToPoint(float xpos, float ypos) {	

		float x = float(xpos) / float(mViewportWidth) - 0.5f;
		float y = float(mViewportHeight - ypos) / float(mViewportHeight) - 0.5f;

		float dx = x - mRegX;
		float dy = y - mRegY;

		Vec3f viewDir = mCameraPos - mCameraTarget;
		Vec3f rightDir = mCameraUp.cross(viewDir).normalize();
		Vec3f upDir = viewDir.cross(rightDir).normalize();
		
		Vec3f t = upDir * -dy + rightDir * -dx;
		mCameraPos += t;
		mCameraTarget += t;
		
		registerPoint(xpos, ypos);
	}

}
