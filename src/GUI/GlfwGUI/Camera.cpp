#include "Camera.h"

#include <glad/glad.h>
#include <iostream>
#include <math.h>

namespace dyno
{
	Camera::Camera() {
		mFocusDist = 3.0f;
		mEyePos = Vec3f(0, 0, mFocusDist);
		mTargetPos = Vec3f(0);
		mLightPos = Vec3f(0, 0, mFocusDist);
		mRotAngle = 0;
		mRotAxis = Vec3f(0, 1, 0);
		mFov = 0.90f;

		mYaw = 0.0f;// M_PI / 2.0f;
		mPitch = 0.0f;//M_PI / 4.0f;
	}


	void Camera::setGL(float neardist, float fardist, float width, float height) {
		float diag = sqrt(width*width + height*height);
		float top = height / diag * 0.5f*mFov*neardist;
		float bottom = -top;
		float right = width / diag* 0.5f*mFov*neardist;
		float left = -right;

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(left, right, bottom, top, neardist, fardist);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glRotatef(180.0f / M_PI*mRotAngle, mRotAxis[0], mRotAxis[1], mRotAxis[2]);
		glTranslatef(-mEyePos[0], -mEyePos[1], -mEyePos[2]);

		//printf("Angle: %f; Axis: %f %f %f \n", mRotAngle, mRotAxis[0], mRotAxis[1], mRotAxis[2]);

		GLfloat pos[] = { mLightPos[0], mLightPos[1], mLightPos[2],1 };
		glLightfv(GL_LIGHT0, GL_POSITION, pos);

		mViewportWidth = (int)width;
		mViewportHeight = (int)height;

		mNear = neardist;
		mFar = fardist;
		mRight = right;
	}

	int Camera::viewportWidth() const {
		return mViewportWidth;
	}

	int Camera::viewportHeight() const {
		return mViewportHeight;
	}

	Vec3f Camera::getEyePos() const {
		return mEyePos;
	}

	void Camera::rotate(float dx, float dy)
	{
		float newYaw = mYaw + dx;
		float newPitch = mPitch + dy;

		Quat1f oldQuat(mRotAngle, mRotAxis);
		oldQuat.w = -oldQuat.w;
		Vec3f curViewdir = oldQuat.rotate(Vec3f(0, 0, -1));

		Vec3f eyeCenter = mEyePos + mFocusDist * curViewdir;
		Vec3f lightCenter = mLightPos + mFocusDist * curViewdir;

		Quat1f newQuat = Quat1f(newPitch, Vec3f(1.0f, 0.0f, 0.0f)) * Quat1f(newYaw, Vec3f(0.0f, 1.0f, 0.0f));

		newQuat.toRotationAxis(mRotAngle, mRotAxis);

		Quat1f q2 = newQuat;
		q2.w = -q2.w;
		Quat1f qFinal = q2;
		//Quat1f qFinal = Quat1f(newPitch, vecX) * q;

		Vec3f newViewdir = q2.rotate(Vec3f(0, 0, -1));

		mEyePos = eyeCenter - mFocusDist * newViewdir;
		mLightPos = lightCenter - mFocusDist * newViewdir;

		mYaw = newYaw;
		mPitch = newPitch;
	}

	Vec3f Camera::getViewDir() const {
		Quat1f q(mRotAngle, mRotAxis);
		q.w = -q.w;
		Vec3f viewdir = q.rotate(Vec3f(0, 0, 1));
		return viewdir;
	}

	void Camera::getCoordSystem(Vec3f &view, Vec3f &up, Vec3f &right) const {
		Quat1f q(mRotAngle, mRotAxis);
		q.w = -q.w;
		view = q.rotate(Vec3f(0, 0, 1));
		up = q.rotate(Vec3f(0, 1, 0));
		right = -view.cross(up);
	}

	void Camera::translate(const Vec3f translation) {
		Quat1f q(mRotAngle, mRotAxis);
		q.w = -q.w;

		Vec3f xax = q.rotate(Vec3f(1, 0, 0));
		Vec3f yax = q.rotate(Vec3f(0, 1, 0));
		Vec3f zax = q.rotate(Vec3f(0, 0, 1));

		mEyePos += translation[0] * xax +
			translation[1] * yax +
			translation[2] * zax;
	}

	void Camera::translateLight(const Vec3f translation) {
		Quat1f q(mRotAngle, mRotAxis);
		q.w = -q.w;
		
		Vec3f xax = q.rotate(Vec3f(1, 0, 0));
		Vec3f yax = q.rotate(Vec3f(0, 1, 0));
		Vec3f zax = q.rotate(Vec3f(0, 0, 1));

		mLightPos += translation[0] * xax +
			translation[1] * yax +
			translation[2] * zax;
	}

	void Camera::zoom(float amount) {
		mFov += amount / 10;
		mFov = std::max(mFov, 0.01f);
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

	void Camera::registerPoint(float x, float y) {
		mRegX = x;
		mRegY = y;
	}

	void Camera::rotateToPoint(float x, float y) {
		float dx = x - mRegX;
		float dy = y - mRegY;
		Quat1f q = getQuaternion(mRegX, mRegY, x, y);
		rotate(dx, -dy);

		registerPoint(x, y);
	}

	void Camera::translateToPoint(float x, float y) {
		float dx = x - mRegX;
		float dy = y - mRegY;
		float dz = 0;
		translate(Vec3f(-dx, -dy, -dz));

		registerPoint(x, y);
	}

	void Camera::translateLightToPoint(float x, float y) {
		float dx = x - mRegX;
		float dy = y - mRegY;
		float dz = 0;
		translateLight(mFocusDist*Vec3f(dx, dy, dz));

		registerPoint(x, y);
	}

}
