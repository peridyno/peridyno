#pragma once
#include <vector>
#include "Quaternion/quaternion.h"
#include "Vector.h"

#define M_PI 3.14159265358979323846

namespace dyno
{
	typedef Quaternion<float> Quat1f;

	class Camera {

	public:
		Camera();
		~Camera() {};

		void registerPoint(float x, float y);
		void rotateToPoint(float x, float y);
		void translateToPoint(float x, float y);
		void translateLightToPoint(float x, float y);
		void zoom(float amount);
		void setGL(float neardist, float fardist, float width, float height);

		Vector3f getViewDir() const;

		float getPixelArea() const;
		int width() const;
		int height() const;
		Vector3f getEye() const;

		void getCoordSystem(Vector3f &view, Vector3f &up, Vector3f &right) const;
		void rotate(Quat1f &rotquat);
		void translate(const Vector3f translation);

	private:
		void translateLight(const Vector3f translation);
		Vector3f getPosition(float x, float y);
		Quat1f getQuaternion(float x1, float y1, float x2, float y2);

	private:
		float m_x;
		float m_y;

		float m_near;
		float m_far;
		float m_right;
		float m_fov;

		float m_rotation;
		
		int m_width;
		int m_height;

		float m_pixelarea;

		Vector3f m_eye;
		Vector3f m_light;
		Vector3f m_rotation_axis;
	};

}

