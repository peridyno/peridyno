#include "OrbitCamera.h"

#include <iostream>
#include <math.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>


namespace dyno
{
	OrbitCamera::OrbitCamera() 
		: Camera()
	{
		mFocusDist = 3.0f;
		mEyePos = Vec3f(0.0f, 0.0f, 3.0f);
		mFov = 0.90f;

		this->setClipFar(20.0f);
	}

	Vec3f OrbitCamera::getEyePos() const {
		return mEyePos;
	}

	Vec3f OrbitCamera::getTargetPos() const
	{
		return mEyePos + mFocusDist * getViewDir();
	}

	void OrbitCamera::setEyePos(const Vec3f& p)
	{
		Quat1f q(mRotAngle, mRotAxis);
		q.w = -q.w;

		//Camera coordinate system
		Vec3f view = q.rotate(Vec3f(0, 0, 1));
		Vec3f up = q.rotate(Vec3f(0, 1, 0));
		Vec3f right = q.rotate(Vec3f(1, 0, 0));

		Vec3f tarPos = getTargetPos();
		
		Vec3f ND = p - tarPos;

		float d = ND.norm();
		ND.normalize();
		if (d > mFocusDistMax) {
			mEyePos = tarPos - mFocusDistMax * ND;
		}
		else {
			mEyePos = p;
		}

		mFocusDist = (mEyePos - tarPos).norm();
		mPitch = acosf(-ND.dot(up)) - M_PI / 2.0f;
		mYaw = -atan2f(ND.dot(right), ND.dot(view));
	}

	void OrbitCamera::setTargetPos(const Vec3f& p)
	{
		Quat1f q(mRotAngle, mRotAxis);
		q.w = -q.w;

		//Camera coordinate system
		Vec3f view = q.rotate(Vec3f(0, 0, 1));
		Vec3f up = q.rotate(Vec3f(0, 1, 0));
		Vec3f right = q.rotate(Vec3f(1, 0, 0));

		Vec3f tarPos = p;

		Vec3f ND = mEyePos - tarPos;

		float d = ND.norm();
		ND.normalize();

		mFocusDist = std::min(mFocusDistMax, (mEyePos - tarPos).norm());
		mPitch = acosf(-ND.dot(up)) - M_PI / 2.0f;
		mYaw = -atan2f(ND.dot(right), ND.dot(view));
	}

	void OrbitCamera::rotate(float dx, float dy)
	{
		float newYaw = mYaw + dx;
		float newPitch = mPitch + dy;

		Quat1f oldQuat = getQuaternion(mYaw, mPitch);
		oldQuat.w = -oldQuat.w;
		Vec3f curViewdir = oldQuat.rotate(Vec3f(0, 0, -1));
		Vec3f eyeCenter = mEyePos + mFocusDist * curViewdir;

		Quat1f newQuat = getQuaternion(newYaw, newPitch);
		Quat1f q2 = newQuat;
		q2.w = -q2.w;
		Quat1f qFinal = q2;
		//Quat1f qFinal = Quat1f(newPitch, vecX) * q;

		Vec3f newViewdir = q2.rotate(Vec3f(0, 0, -1));

		mEyePos = eyeCenter - mFocusDist * newViewdir;

		mYaw = newYaw;
		mPitch = newPitch;
	}

	Vec3f OrbitCamera::getViewDir() const {
		Quat1f q = getQuaternion(mYaw, mPitch);
		q.w = -q.w;

		Vec3f viewdir = q.rotate(Vec3f(0, 0, -1));
		return viewdir;
	}

	void OrbitCamera::getCoordSystem(Vec3f &view, Vec3f &up, Vec3f &right) const {
		Quat1f q = getQuaternion(mYaw, mPitch);
		q.w = -q.w;

		view = q.rotate(Vec3f(0, 0, -1));
		up = q.rotate(Vec3f(0, 1, 0));
		right = view.cross(up);
	}

	void OrbitCamera::translate(const Vec3f translation) {
		Quat1f q = getQuaternion(mYaw, mPitch);
		q.w = -q.w;

		Vec3f xax = q.rotate(Vec3f(1, 0, 0));
		Vec3f yax = q.rotate(Vec3f(0, 1, 0));
		Vec3f zax = q.rotate(Vec3f(0, 0, 1));

		mEyePos += translation[0] * xax +
			translation[1] * yax +
			translation[2] * zax;
	}

	void OrbitCamera::zoom(float amount) {
		Quat1f oldQuat = getQuaternion(mYaw, mPitch);

		// calculate the view direction
		//Quat1f oldQuat(mRotAngle, mRotAxis);
		oldQuat.w = -oldQuat.w;
		Vec3f curViewdir = oldQuat.rotate(Vec3f(0, 0, -1));

		Vec3f eyeCenter = mEyePos + mFocusDist * curViewdir;

		float logDist = std::log10(mFocusDist);
		float logMin = std::log10(mFocusDistMin);
		float logMax = std::log10(mFocusDistMax);
		float frac = (logDist - logMax) / (logMax - logMin);

		mFocusDist += mZoomSpeed * amount * std::pow(10.0f, frac);
		mFocusDist = std::min(std::max(mFocusDist, mFocusDistMin), mFocusDistMax);
		mEyePos = eyeCenter - mFocusDist * curViewdir;
	}

	Vec3f OrbitCamera::getPosition(float x, float y) {
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

	Quat1f OrbitCamera::getQuaternion(float x1, float y1, float x2, float y2) {
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

	Quat1f OrbitCamera::getQuaternion(float yaw, float pitch) const
	{
		Quat1f oldQuat = Quat1f(pitch, Vec3f(1.0f, 0.0f, 0.0f)) * Quat1f(yaw, Vec3f(0.0f, 1.0f, 0.0f));
		oldQuat.toRotationAxis(float(mRotAngle), Vec3f(mRotAxis));

		return oldQuat;
	}

	void OrbitCamera::registerPoint(float x, float y) {
// 		mRegX = x;
// 		mRegY = y;
		mRegX = float(x) / float(mViewportWidth);
		mRegY = float(mViewportHeight - y) / float(mViewportHeight);
	}

	glm::mat4 OrbitCamera::getViewMat()
	{
		Vec3f upDir;// = Vec3f(0, 1, 0);
		Vec3f viewDir;// = getViewDir();
		Vec3f rightDir;// = upDir.cross(viewDir).normalize();

		getCoordSystem(viewDir, upDir, rightDir);
		Vec3f targetPos = mEyePos + mFocusDist * viewDir;

		return glm::lookAt(mEyePos.data_ * mDistanceUnit, targetPos.data_ * mDistanceUnit, upDir.data_);
	}

	glm::mat4 OrbitCamera::getProjMat()
	{
		float aspect = std::max(float(mViewportWidth), 1.0f) / std::max(float(mViewportHeight), 1.0f);
		
		glm::mat4 projection;

		if (mProjectionType == Perspective)
		{
			projection = glm::perspective(mFov, aspect, mNear * mDistanceUnit, mFar * mDistanceUnit);
		}
		else
		{
			float half_depth = (mEyePos - mTargetPos).norm() * mDistanceUnit;
			projection = glm::ortho(-half_depth * aspect, half_depth * aspect, -half_depth, half_depth, -5.0f * half_depth, 5.0f* half_depth);
		}
			
		return projection;
	}

	

	void OrbitCamera::rotateToPoint(float x, float y) {
		float tx = float(x) / float(mViewportWidth);
		float ty = float(mViewportHeight - y) / float(mViewportHeight);

		float dx = tx - mRegX;
		float dy = ty - mRegY;
		Quat1f q = getQuaternion(mRegX, mRegY, tx, ty);
		rotate(mSpeed*dx, -mSpeed * dy);

		registerPoint(x, y);
	}

	void OrbitCamera::translateToPoint(float x, float y) {
		float tx = float(x) / float(mViewportWidth);
		float ty = float(mViewportHeight - y) / float(mViewportHeight);

		float dx = tx - mRegX;
		float dy = ty - mRegY;
		float dz = 0;
		translate(mSpeed*Vec3f(-dx, -dy, -dz));

		registerPoint(x, y);
	}
}
