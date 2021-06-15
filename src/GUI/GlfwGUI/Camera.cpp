#include "Camera.h"

#include <glad/glad.h>
#include <iostream>
#include <math.h>

#include <glm/gtc/matrix_transform.hpp>

namespace dyno
{
	Camera::Camera() {

		mFocusDist = 3.0f;
		mEyePos = Vec3f(0, 0, mFocusDist);
		mTargetPos = Vec3f(0);

		mRotAngle = 0;
		mRotAxis = Vec3f(0, 1, 0);
		mFov = glm::radians(45.0f);

		mYaw = 0.0f;// M_PI / 2.0f;
		mPitch = 0.0f;//M_PI / 4.0f;
	}

	void Camera::reset()
	{
		// TODO: reset to fit scene
	}
	
	glm::mat4 Camera::getViewMat()
	{
		glm::mat4 view = glm::rotate(glm::mat4(), 
			(float)(mRotAngle), 
			glm::vec3(mRotAxis[0], mRotAxis[1], mRotAxis[2]));

		view =  glm::translate(view, glm::vec3(-mEyePos[0], -mEyePos[1], -mEyePos[2])) ;

		return view;
	}

	glm::mat4 Camera::getProjMat()
	{
		return glm::perspective(mFov, float(mViewportWidth) / mViewportHeight, mNear, mFar);
	}

	void Camera::rotate(float dx, float dy)
	{
		float newYaw = mYaw + dx;
		float newPitch = mPitch + dy;

		Quat1f oldQuat(mRotAngle, mRotAxis);
		oldQuat.w = -oldQuat.w;

		Vec3f curViewdir = mEyePos - mTargetPos;
		curViewdir.normalize();//oldQuat.rotate(Vec3f(0, 0, -1));
		Vec3f eyeCenter = mTargetPos;//mEyePos + mFocusDist * curViewdir;

		Quat1f newQuat = Quat1f(newPitch, Vec3f(1.0f, 0.0f, 0.0f)) * Quat1f(newYaw, Vec3f(0.0f, 1.0f, 0.0f));

		newQuat.toRotationAxis(mRotAngle, mRotAxis);

		Quat1f q2 = newQuat;
		q2.w = -q2.w;
		Quat1f qFinal = q2;

		Vec3f newViewdir = q2.rotate(Vec3f(0, 0, -1));

		mEyePos = eyeCenter - mFocusDist * newViewdir;
		mYaw = newYaw;
		mPitch = newPitch;
	}


	void Camera::translate(const Vec3f translation) {
		Quat1f q(mRotAngle, mRotAxis);
		q.w = -q.w;

		Vec3f xax = q.rotate(Vec3f(1, 0, 0));
		Vec3f yax = q.rotate(Vec3f(0, 1, 0));
		Vec3f zax = q.rotate(Vec3f(0, 0, 1));

		Vec3f t = translation[0] * xax +
			translation[1] * yax +
			translation[2] * zax;
		mEyePos += t;
		mTargetPos += t;
	}


	void Camera::zoom(float amount) {
		mFov += amount / 10;
		mFov = std::max(mFov, 0.05f);
	}

	Vec3f Camera::getPosition(float x, float y) {
		float r = x*x + y*y;
		float t = 0.5f * 1 * 1;
		if (r < t) {
			Vec3f result(x, y, sqrt(2.0f*t - r));
			result.normalize();
			return result;
		}
		else {
			Vec3f result(x, y, t / sqrt(r));
			result.normalize();
			return result;
		}
	}

	Quat1f Camera::getQuaternion(float x1, float y1, float x2, float y2) {
		if ((x1 == x2) && (y1 == y2)) {
			return Quat1f();
		}
		Vec3f pos1 = getPosition(x1, y1);
		Vec3f pos2 = getPosition(x2, y2);
		Vec3f rotaxis = pos1.cross(pos2);
		rotaxis.normalize();
		float rotangle = 2 * sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
		return Quat1f(rotangle, rotaxis);
	}

	void Camera::registerPoint(float xpos, float ypos) {
		mRegX = float(xpos) / float(mViewportWidth) - 0.5f;
		mRegY = float(mViewportHeight - ypos) / float(mViewportHeight) - 0.5f;
	}

	void Camera::rotateToPoint(float xpos, float ypos) {
		float x = float(xpos) / float(mViewportWidth) - 0.5f;
		float y = float(mViewportHeight - ypos) / float(mViewportHeight) - 0.5f;

		float dx = x - mRegX;
		float dy = y - mRegY;
		Quat1f q = getQuaternion(mRegX, mRegY, x, y);
		rotate(dx, -dy);
		registerPoint(xpos, ypos);
	}

	void Camera::translateToPoint(float xpos, float ypos) {
		
		float x = float(xpos) / float(mViewportWidth) - 0.5f;
		float y = float(mViewportHeight - ypos) / float(mViewportHeight) - 0.5f;

		float dx = x - mRegX;
		float dy = y - mRegY;
		float dz = 0;
		translate(Vec3f(-dx, -dy, -dz));
		registerPoint(xpos, ypos);
	}

}
