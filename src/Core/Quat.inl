
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "Vector.h"
#include "Matrix.h"

namespace dyno
{
	template <typename Real>
	DYN_FUNC Quat<Real>::Quat() :
		w(1),
		x(0),
		y(0),
		z(0)
	{
	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(Real _x, Real _y, Real _z, Real _w) :
		w(_w),
		x(_x),
		y(_y),
		z(_z)
	{
	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(Real rot, const Vector<Real, 3>& axis)
	{
		const Real a = rot * Real(0.5);
		const Real s = glm::sin(a);
		w = glm::cos(a);
		x = axis[0] * s;
		y = axis[1] * s;
		z = axis[2] * s;
	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(const Quat<Real> & quat) :
		w(quat.w),
		y(quat.y),
		z(quat.z),
		x(quat.x)
	{

	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(const Real yaw, const Real pitch, const Real roll)
	{
		Real cy = glm::cos(Real(yaw * 0.5));
		Real sy = glm::sin(Real(yaw * 0.5));
		Real cp = glm::cos(Real(pitch * 0.5));
		Real sp = glm::sin(Real(pitch * 0.5));
		Real cr = glm::cos(Real(roll * 0.5));
		Real sr = glm::sin(Real(roll * 0.5));

		w = cr * cp * cy + sr * sp * sy;
		x = sr * cp * cy - cr * sp * sy;
		y = cr * sp * cy + sr * cp * sy;
		z = cr * cp * sy - sr * sp * cy;
	}


	template <typename Real>
	DYN_FUNC Quat<Real> & Quat<Real>::operator = (const Quat<Real> &quat)
	{
		w = quat.w;
		x = quat.x;
		y = quat.y;
		z = quat.z;
		return *this;
	}

	template <typename Real>
	DYN_FUNC Quat<Real> & Quat<Real>::operator += (const Quat<Real> &quat)
	{
		w += quat.w;
		x += quat.x;
		y += quat.y;
		z += quat.z;
		return *this;
	}

	template <typename Real>
	DYN_FUNC Quat<Real> & Quat<Real>::operator -= (const Quat<Real> &quat)
	{
		w -= quat.w;
		x -= quat.x;
		y -= quat.y;
		z -= quat.z;
		return *this;
	}

	template <typename Real>
	DYN_FUNC Quat<Real>  Quat<Real>::operator - (const Quat<Real> &quat) const
	{
		return Quat(x - quat.x, y - quat.y, z - quat.z, w - quat.w);
	}

	template <typename Real>
	DYN_FUNC Quat<Real>  Quat<Real>::operator - (void) const
	{
		return Quat(-x, -y, -z, -w);
	}

	template <typename Real>
	DYN_FUNC Quat<Real>  Quat<Real>::operator + (const Quat<Real> &quat) const
	{
		return Quat(x + quat.x, y + quat.y, z + quat.z, w + quat.w);
	}

	template <typename Real>
	DYN_FUNC Quat<Real>  Quat<Real>::operator * (const Real& scale) const
	{
		return Quat(x * scale, y * scale, z * scale, w * scale);
	}

	template <typename Real>
	DYN_FUNC Quat<Real> Quat<Real>::operator * (const Quat<Real>& q) const
	{
		Quat result;
		
		result.w = -x * q.x - y * q.y - z * q.z + w * q.w;

		result.x = x * q.w + y * q.z - z * q.y + w * q.x;
		result.y = -x * q.z + y * q.w + z * q.x + w * q.y;
		result.z = x * q.y - y * q.x + z * q.w + w * q.z;
		
		return result;
	}

	template <typename Real>
	DYN_FUNC Quat<Real>  Quat<Real>::operator / (const Real& scale) const
	{
		return Quat(x / scale, y / scale, z / scale, w / scale);
	}

	template <typename Real>
	DYN_FUNC bool  Quat<Real>::operator == (const Quat<Real> &quat) const
	{
		if (w == quat.w && x == quat.x && y == quat.y && z == quat.z)
			return true;
		return false;
	}

	template <typename Real>
	DYN_FUNC bool  Quat<Real>::operator != (const Quat<Real> &quat) const
	{
		if (*this == quat)
			return false;
		return true;
	}

	template <typename Real>
	DYN_FUNC Real Quat<Real>::norm() const
	{
		Real result = w * w + x * x + y * y + z * z;
		result = glm::sqrt(result);

		return result;
	}

	template <typename Real>
	DYN_FUNC Real Quat<Real>::normSquared() const
	{
		return w * w + x * x + y * y + z * z;
	}

	template <typename Real>
	DYN_FUNC Quat<Real>& Quat<Real>::normalize()
	{
		Real d = norm();
		// Set the rotation along the x-axis
		if (d < 0.00001) {
			z = Real(1.0);
			x = y = w = Real(0.0);
			return *this;
		}
		d = Real(1) / d;
		x *= d;
		y *= d;
		z *= d;
		w *= d;
		return *this;
	}

	template <typename Real>
	DYN_FUNC Quat<Real> Quat<Real>::inverse() const
	{
		return conjugate() / normSquared();
	}

	template <typename Real>
	DYN_FUNC void Quat<Real>::toEulerAngle(Real& yaw, Real& pitch, Real& roll) const
	{
		// roll (x-axis rotation)
		Real sinr_cosp = 2 * (w * x + y * z);
		Real cosr_cosp = 1 - 2 * (x * x + y * y);
		roll = atan2(sinr_cosp, cosr_cosp);

		// pitch (y-axis rotation)
		Real sinp = 2 * (w * y - z * x);
		if (glm::abs(sinp) >= 1)
			pitch = sinp > 0 ? Real(M_PI / 2) : -Real(M_PI / 2); // use 90 degrees if out of range
		else
			pitch = glm::asin(sinp);

		// yaw (z-axis rotation)
		Real siny_cosp = 2 * (w * z + x * y);
		Real cosy_cosp = 1 - 2 * (y * y + z * z);
		yaw = atan2(siny_cosp, cosy_cosp);
	}

	template <typename Real>
	DYN_FUNC Real Quat<Real>::angle() const
	{
		return glm::acos(w) * (Real)(2);
	}

	template <typename Real>
	DYN_FUNC Real Quat<Real>::angle(const Quat<Real>& quat) const
	{
		return glm::acos(dot(quat)) * (Real)(2);
	}

	template <typename Real>
	DYN_FUNC Real Quat<Real>::dot(const Quat<Real> & quat) const
	{
		return w * quat.w + x * quat.x + y * quat.y + z * quat.z;
	}

	template <typename Real>
	DYN_FUNC Quat<Real> Quat<Real>::conjugate() const
	{
		return Quat<Real>(-x, -y, -z, w);
	}

	// Refer to "https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles" for more details
	template <typename Real>
	DYN_FUNC Vector<Real, 3> Quat<Real>::rotate(const Vector<Real, 3>& v) const
	{
		// Extract the vector part of the quaternion
		Vector<Real, 3> u(x, y, z);

		// Extract the scalar part of the quaternion
		Real s = w;

		// Do the math
		return    2.0f * u.dot(v) * u
				+ (s*s - u.dot(u)) * v
				+ 2.0f * s * u.cross(v);
	}


	template <typename Real>
	DYN_FUNC SquareMatrix<Real, 3> Quat<Real>::toMatrix3x3() const
	{
		Real x2 = x + x, y2 = y + y, z2 = z + z;
		Real xx = x2 * x, yy = y2 * y, zz = z2 * z;
		Real xy = x2 * y, xz = x2 * z, xw = x2 * w;
		Real yz = y2 * z, yw = y2 * w, zw = z2 * w;
		return SquareMatrix<Real, 3>(Real(1) - yy - zz, xy - zw, xz + yw,
			xy + zw, Real(1) - xx - zz, yz - xw,
			xz - yw, yz + xw, Real(1) - xx - yy);
	}

	template <typename Real>
	DYN_FUNC SquareMatrix<Real, 4> Quat<Real>::toMatrix4x4() const
	{
		Real x2 = x + x, y2 = y + y, z2 = z + z;
		Real xx = x2 * x, yy = y2 * y, zz = z2 * z;
		Real xy = x2 * y, xz = x2 * z, xw = x2 * w;
		Real yz = y2 * z, yw = y2 * w, zw = z2 * w;
		Real entries[16];
		entries[0] = Real(1) - yy - zz;
		entries[1] = xy - zw;
		entries[2] = xz + yw,
			entries[3] = 0;
		entries[4] = xy + zw;
		entries[5] = Real(1) - xx - zz;
		entries[6] = yz - xw;
		entries[7] = 0;
		entries[8] = xz - yw;
		entries[9] = yz + xw;
		entries[10] = Real(1) - xx - yy;
		entries[11] = 0;
		entries[12] = 0;
		entries[13] = 0;
		entries[14] = 0;
		entries[15] = 1;
		return SquareMatrix<Real, 4>(entries[0], entries[1], entries[2], entries[3],
			entries[4], entries[5], entries[6], entries[7],
			entries[8], entries[9], entries[10], entries[11],
			entries[12], entries[13], entries[14], entries[15]);

	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(const SquareMatrix<Real, 3>& matrix)
	{
		Real tr = matrix(0, 0) + matrix(1, 1) + matrix(2, 2);
		if (tr > 0.0)
		{
			Real s = glm::sqrt(tr + Real(1.0));
			w = s * Real(0.5);
			if (s != 0.0)
				s = Real(0.5) / s;
			x = s * (matrix(2, 1) - matrix(1, 2));
			y = s * (matrix(0, 2) - matrix(2, 0));
			z = s * (matrix(1, 0) - matrix(0, 1));
		}
		else
		{
			int i = 0, j, k;
			int next[3] = { 1, 2, 0 };
			Real q[4];
			if (matrix(1, 1) > matrix(0, 0)) i = 1;
			if (matrix(2, 2) > matrix(i, i)) i = 2;
			j = next[i];
			k = next[j];
			Real s = glm::sqrt(matrix(i, i) - matrix(j, j) - matrix(k, k) + Real(1.0));
			q[i] = s * Real(0.5);
			if (s != 0.0)
				s = Real(0.5) / s;
			q[3] = s * (matrix(k, j) - matrix(j, k));
			q[j] = s * (matrix(j, i) + matrix(i, j));
			q[k] = s * (matrix(k, i) + matrix(i, k));
			x = q[0];
			y = q[1];
			z = q[2];
			w = q[3];
		}
	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(const SquareMatrix<Real, 4>& matrix)
	{
		Real tr = matrix(0, 0) + matrix(1, 1) + matrix(2, 2);
		if (tr > 0.0)
		{
			Real s = glm::sqrt(tr + Real(1.0));
			w = s * Real(0.5);
			if (s != 0.0)
				s = Real(0.5) / s;
			x = s * (matrix(2, 1) - matrix(1, 2));
			y = s * (matrix(0, 2) - matrix(2, 0));
			z = s * (matrix(1, 0) - matrix(0, 1));
		}
		else
		{
			int i = 0, j, k;
			int next[3] = { 1, 2, 0 };
			Real q[4];
			if (matrix(1, 1) > matrix(0, 0)) i = 1;
			if (matrix(2, 2) > matrix(i, i)) i = 2;
			j = next[i];
			k = next[j];
			Real s = glm::sqrt(matrix(i, i) - matrix(j, j) - matrix(k, k) + Real(1.0));
			q[i] = s * Real(0.5);
			if (s != 0.0)
				s = Real(0.5) / s;
			q[3] = s * (matrix(k, j) - matrix(j, k));
			q[j] = s * (matrix(j, i) + matrix(i, j));
			q[k] = s * (matrix(k, i) + matrix(i, k));
			x = q[0];
			y = q[1];
			z = q[2];
			w = q[3];
		}
	}

	template <typename Real>
	DYN_FUNC void Quat<Real>::toRotationAxis(Real& rot, Vector<Real, 3>& axis) const
	{
		rot = Real(2) * glm::acos(w);
		if (glm::abs(rot) < EPSILON) {
			axis[0] = Real(0); axis[1] = Real(0); axis[2] = Real(1);
			return;
		}
		axis[0] = x;
		axis[1] = y;
		axis[2] = z;
		axis.normalize();
	}

	//template class Quat<float>;
	//template class Quat<double>;
	////convenient typedefs
	//typedef Quat<float> Quat1f;
	//typedef Quat<double> Quat1d;
}
