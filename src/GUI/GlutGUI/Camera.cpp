/******************************************************************************
Copyright (c) 2007 Bart Adams (bart.adams@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software. The authors shall be
acknowledged in scientific publications resulting from using the Software
by referencing the ACM SIGGRAPH 2007 paper "Adaptively Sampled Particle
Fluids".

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
******************************************************************************/

#include <GL/gl.h>
#include "Camera.h"
#include <iostream>
#include <math.h>
using namespace std;

namespace dyno
{
	Camera::Camera() {
		m_eye = Vector3f(0, 0, 3);
		m_light = Vector3f(0, 0, 3);
		m_rotation = 0;
		m_rotation_axis = Vector3f(0, 1, 0);
		m_fov = 0.90f;
	}


	void Camera::setGL(float neardist, float fardist, float width, float height) {
		float diag = sqrt(width*width + height*height);
		float top = height / diag * 0.5f*m_fov*neardist;
		float bottom = -top;
		float right = width / diag* 0.5f*m_fov*neardist;
		float left = -right;

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(left, right, bottom, top, neardist, fardist);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glRotatef(180.0f / M_PI*m_rotation, m_rotation_axis[0], m_rotation_axis[1], m_rotation_axis[2]);
		glTranslatef(-m_eye[0], -m_eye[1], -m_eye[2]);

		GLfloat pos[] = { m_light[0], m_light[1], m_light[2],1 };
		glLightfv(GL_LIGHT0, GL_POSITION, pos);

		m_width = (int)width;
		m_height = (int)height;
		m_pixelarea = 4 * right*top / (width*height);
		m_near = neardist;
		m_far = fardist;
		m_right = right;
	}

	int Camera::width() const {
		return m_width;
	}

	int Camera::height() const {
		return m_height;
	}

	float Camera::getPixelArea() const {
		return m_pixelarea;
	}

	Vector3f Camera::getEye() const {
		return m_eye;
	}

	void Camera::rotate(Quat1f &rotquat) {
		// set up orthogonal camera system
		Quat1f q(m_rotation, m_rotation_axis);
		//q.x = -q.x;
		q.setX(-q.x());
		Vector3f viewdir(0, 0, -1);
		q.rotateVector(viewdir);
		// end set up orth system
		//   q = Quat1f(angle, axis);
		q = rotquat;
		Quat1f currq(m_rotation, m_rotation_axis);
		Vector3f rotcenter = m_eye + 3.0f*viewdir;
		Vector3f rotcenter2 = m_light + 3.0f*viewdir;
// 		currq = q.ComposeWith(currq);
// 		currq.ToRotAxis(m_rotation, m_rotation_axis);
		currq = q * currq;
		q.normalize();
		currq.toRotationAxis(m_rotation, m_rotation_axis);
		// set up orthogonal camera system
		Quat1f q2(m_rotation, m_rotation_axis);
		//q2.x = -q2.x;
		q2.setX(-q2.x());
		Vector3f viewdir2(0, 0, -1);
		q2.rotateVector(viewdir2);

		m_eye = rotcenter - 3.0f*viewdir2;
		m_light = rotcenter2 - 3.0f*viewdir2;
	}

	Vector3f Camera::getViewDir() const {
		Quat1f q(m_rotation, m_rotation_axis);
		//q.x = -q.x;
		q.setX(-q.x());
		Vector3f viewdir(0, 0, 1);
		q.rotateVector(viewdir);
		return viewdir;
	}

	void Camera::getCoordSystem(Vector3f &view, Vector3f &up, Vector3f &right) const {
		Quat1f q(m_rotation, m_rotation_axis);
		//q.x = -q.x;
		q.setX(-q.x());
		view = Vector3f(0, 0, 1);
		q.rotateVector(view);
		up = Vector3f(0, 1, 0);
		q.rotateVector(up);
		right = -view.cross(up);
	}

	void Camera::translate(const Vector3f translation) {
		Quat1f q(m_rotation, m_rotation_axis);
		//q.x = -q.x;
		q.setX(-q.x());
		Vector3f xax(1, 0, 0);
		Vector3f yax(0, 1, 0);
		Vector3f zax(0, 0, 1);

		q.rotateVector(xax);
		q.rotateVector(yax);
		q.rotateVector(zax);

		m_eye += translation[0] * xax +
			translation[1] * yax +
			translation[2] * zax;
	}

	void Camera::translateLight(const Vector3f translation) {
		Quat1f q(m_rotation, m_rotation_axis);
		//q.x = -q.x;
		q.setX(-q.x());
		Vector3f xax(1, 0, 0);
		Vector3f yax(0, 1, 0);
		Vector3f zax(0, 0, 1);

		q.rotateVector(xax);
		q.rotateVector(yax);
		q.rotateVector(zax);

		m_light += translation[0] * xax +
			translation[1] * yax +
			translation[2] * zax;
	}

	void Camera::zoom(float amount) {
		m_fov += amount / 10;
		m_fov = max(m_fov, 0.01f);
	}

	Vector3f Camera::getPosition(float x, float y) {
		float r = x*x + y*y;
		float t = 0.5f * 1 * 1;
		if (r < t) {
			Vector3f result(x, y, sqrt(2.0f*t - r));
			result.normalize();
			return result;
		}
		else {
			Vector3f result(x, y, t / sqrt(r));
			result.normalize();
			return result;
		}
	}

	Quat1f Camera::getQuaternion(float x1, float y1, float x2, float y2) {
		if ((x1 == x2) && (y1 == y2)) {
			return Quat1f(1, 0, 0, 0);
		}
		Vector3f pos1 = getPosition(x1, y1);
		Vector3f pos2 = getPosition(x2, y2);
		Vector3f rotaxis = pos1.cross(pos2);
		rotaxis.normalize();
		float rotangle = 2 * sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
		return Quat1f(rotangle, rotaxis);
	}

	void Camera::registerPoint(float x, float y) {
		m_x = x;
		m_y = y;
	}
	void Camera::rotateToPoint(float x, float y) {
		Quat1f q = getQuaternion(m_x, m_y, x, y);
		registerPoint(x, y);
		rotate(q);
	}
	void Camera::translateToPoint(float x, float y) {
		float dx = x - m_x;
		float dy = y - m_y;
		float dz = 0;
		registerPoint(x, y);
		translate(Vector3f(-dx, -dy, -dz));
	}

	void Camera::translateLightToPoint(float x, float y) {
		float dx = x - m_x;
		float dy = y - m_y;
		float dz = 0;
		registerPoint(x, y);
		translateLight(Vector3f(3 * dx, 3 * dy, 3 * dz));
	}

}
