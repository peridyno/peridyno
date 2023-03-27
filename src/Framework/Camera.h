#pragma once

#include <glm/glm.hpp>

// TODO: we prefer not depend on Core library
#include <Vector.h>
#include <Quat.h>

#include "Primitive/Primitive3D.h"

namespace dyno
{
	typedef Quat<float> Quat1f;

	class Camera {
	public:
		Camera() {};
		~Camera() {};

		enum ProjectionType
		{
			Perspective,
			Orthogonal
		};

		virtual glm::mat4 getViewMat() = 0;
		virtual glm::mat4 getProjMat() = 0;

		virtual void rotateToPoint(float x, float y) = 0;
		virtual void translateToPoint(float x, float y) = 0;
		virtual void zoom(float amount) = 0;

		virtual void registerPoint(float x, float y) = 0;

		void setWidth(int width) { mViewportWidth = width; }
		void setHeight(int height) { mViewportHeight = height; }
		
		void setClipNear(float zNear) { mNear = zNear; }
		void setClipFar(float zFar) { mFar = zFar; }

		int viewportWidth() const {	return mViewportWidth; }
		int viewportHeight() const { return mViewportHeight; }

		float clipNear() const { return mNear; }
		float clipFar() const { return mFar; }

		virtual void setEyePos(const Vec3f& p) = 0;
		virtual void setTargetPos(const Vec3f& p) = 0;

		virtual Vec3f getEyePos() const = 0;
		virtual Vec3f getTargetPos() const = 0;

		TRay3D<float> castRayInWorldSpace(float x, float y);

		void setUnitScale(float unit) { mUnitScale = unit; }

		float unitScale() { return mUnitScale; }
		
		void setProjectionType(ProjectionType type) { mProjectionType = type; }
		ProjectionType projectionType() { return mProjectionType; }

	protected:
		float mNear = 0.01f;
		float mFar = 10.0f;
		float mFov = 0.0f;

		int mViewportWidth;
		int mViewportHeight;

		ProjectionType mProjectionType = Perspective;	//0:pers 1:ortho

		//Distance unit
		float mUnitScale = 1.0f;
	};

}

