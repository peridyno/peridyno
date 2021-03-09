#include <cmath>
#include <cstdlib>
#include <iostream>
#include "Vector.h"
#include "Matrix.h"

namespace dyno
{
	template <typename Real>
	DYN_FUNC Quat<Real>::Quat() :
		x_(0),
		y_(0),
		z_(0),
		w_(1)
	{
	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(Real x, Real y, Real z, Real w) :
		x_(x),
		y_(y),
		z_(z),
		w_(w)
	{
	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(const Vector<Real, 3> & unit_axis, Real angle_rad)
	{
		const Real a = angle_rad * (Real)0.5;
		const Real s = glm::sin(a);
		w_ = glm::cos(a);
		x_ = unit_axis[0] * s;
		y_ = unit_axis[1] * s;
		z_ = unit_axis[2] * s;
	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(Real angle_rad, const Vector<Real, 3> & unit_axis)
	{
		Real c = glm::cos(0.5f*angle_rad);
		Real s = glm::sin(0.5f*angle_rad);
		Real t = s / unit_axis.norm();
		x_ = c;
		y_ = unit_axis[0] * t;
		z_ = unit_axis[1] * t;
		w_ = unit_axis[2] * t;
	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(const Real *ptrq) :
		x_(ptrq[0]),
		y_(ptrq[1]),
		z_(ptrq[2]),
		w_(ptrq[3])
	{
	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(const Quat<Real> & quat) :
		x_(quat.x()),
		y_(quat.y()),
		z_(quat.z()),
		w_(quat.w())
	{

	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(const Vector<Real, 3>& euler_angle)
	{
		Real cos_roll = glm::cos(euler_angle[0] * Real(0.5));
		Real sin_roll = glm::sin(euler_angle[0] * Real(0.5));
		Real cos_pitch = glm::cos(euler_angle[1] * Real(0.5));
		Real sin_pitch = glm::sin(euler_angle[1] * Real(0.5));
		Real cos_yaw = glm::cos(euler_angle[2] * Real(0.5));
		Real sin_yaw = glm::sin(euler_angle[2] * Real(0.5));

		w_ = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw;
		x_ = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw;
		y_ = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw;
		z_ = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw;
	}


	template <typename Real>
	DYN_FUNC Quat<Real> & Quat<Real>::operator = (const Quat<Real> &quat)
	{
		w_ = quat.w();
		x_ = quat.x();
		y_ = quat.y();
		z_ = quat.z();
		return *this;
	}

	template <typename Real>
	DYN_FUNC Quat<Real> & Quat<Real>::operator += (const Quat<Real> &quat)
	{
		w_ += quat.w();
		x_ += quat.x();
		y_ += quat.y();
		z_ += quat.z();
		return *this;
	}

	template <typename Real>
	DYN_FUNC Quat<Real> & Quat<Real>::operator -= (const Quat<Real> &quat)
	{
		w_ -= quat.w();
		x_ -= quat.x();
		y_ -= quat.y();
		z_ -= quat.z();
		return *this;
	}

	template <typename Real>
	DYN_FUNC Quat<Real>  Quat<Real>::operator - (const Quat<Real> &quat) const
	{
		return Quat(x_ - quat.x(), y_ - quat.y(), z_ - quat.z(), w_ - quat.w());
	}

	template <typename Real>
	DYN_FUNC Quat<Real>  Quat<Real>::operator - (void) const
	{
		return Quat(-x_, -y_, -z_, -w_);
	}

	template <typename Real>
	DYN_FUNC Quat<Real>  Quat<Real>::operator + (const Quat<Real> &quat) const
	{
		return Quat(x_ + quat.x(), y_ + quat.y(), z_ + quat.z(), w_ + quat.w());
	}

	template <typename Real>
	DYN_FUNC Quat<Real>  Quat<Real>::operator * (const Real& scale) const
	{
		return Quat(x_ * scale, y_ * scale, z_ * scale, w_ * scale);
	}

	template <typename Real>
	DYN_FUNC Quat<Real> Quat<Real>::operator * (const Quat<Real>& q) const
	{
		Quat result;
		result.x_ = x_ * q.x_ - y_ * q.y_ - z_ * q.z_ - w_ * q.w_;
		result.y_ = x_ * q.y_ + y_ * q.x_ + z_ * q.w_ - w_ * q.z_;
		result.z_ = x_ * q.z_ + z_ * q.x_ + w_ * q.y_ - y_ * q.w_;
		result.w_ = x_ * q.w_ + w_ * q.x_ + y_ * q.z_ - z_ * q.y_;
		return result;
	}
	template <typename Real>
	DYN_FUNC Quat<Real> Quat<Real>::multiply_q(const Quat<Real>& q)
	{
		return Quat(w_ * q.x() + x_ * q.w() + y_ * q.z() - z_ * q.y(),
			w_ * q.y() + y_ * q.w() + z_ * q.x() - x_ * q.z(),
			w_ * q.z() + z_ * q.w() + x_ * q.y() - y_ * q.x(),
			w_ * q.w() - x_ * q.x() - y_ * q.y() - z_ * q.z());
	}

	template <typename Real>
	DYN_FUNC Quat<Real>  Quat<Real>::operator / (const Real& scale) const
	{
		return Quat(x_ / scale, y_ / scale, z_ / scale, w_ / scale);
	}

	template <typename Real>
	DYN_FUNC bool  Quat<Real>::operator == (const Quat<Real> &quat) const
	{
		if (w_ == quat.w() && x_ == quat.x() && y_ == quat.y() && z_ == quat.z())
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
	DYN_FUNC Real&  Quat<Real>::operator[] (unsigned int idx)
	{
		switch (idx)
		{
		case 0:
			return x_;
		case 1:
			return y_;
		case 2:
			return z_;
		case 3:
			return w_;
		default:
			return w_;
		}
	}

	template <typename Real>
	DYN_FUNC const Real&  Quat<Real>::operator[] (unsigned int idx) const
	{
		switch (idx)
		{
		case 0:
			return x_;
		case 1:
			return y_;
		case 2:
			return z_;
		case 3:
			return w_;
		default:
			return w_;
		}
	}

	template <typename Real>
	DYN_FUNC Real Quat<Real>::norm()
	{
		Real result = w_ * w_ + x_ * x_ + y_ * y_ + z_ * z_;
		result = glm::sqrt(result);
		return result;
	}

	template <typename Real>
	DYN_FUNC Quat<Real>& Quat<Real>::normalize()
	{
		Real d = glm::sqrt(x_*x_ + y_ * y_ + z_ * z_ + w_ * w_);
		if (d < 0.00001) {
			x_ = 1.0f;
			y_ = z_ = w_ = 0.0f;
			return *this;
		}
		d = Real(1) / d;
		x_ *= d;
		y_ *= d;
		z_ *= d;
		w_ *= d;
		return *this;
	}

	template <typename Real>
	DYN_FUNC void Quat<Real>::set(const Vector<Real, 3>& vec3, Real scale)
	{
		w_ = scale;
		x_ = vec3[0];
		y_ = vec3[1];
		z_ = vec3[2];
	}

	template <typename Real>
	DYN_FUNC void Quat<Real>::set(Real scale, const Vector<Real, 3>& vec3)
	{
		w_ = scale;
		x_ = vec3[0];
		y_ = vec3[1];
		z_ = vec3[2];
	}

	template <typename Real>
	DYN_FUNC void Quat<Real>::set(const Vector<Real, 3>& euler_angle)
	{
		Real cos_roll = glm::cos(euler_angle[0] * Real(0.5));
		Real sin_roll = glm::sin(euler_angle[0] * Real(0.5));
		Real cos_pitch = glm::cos(euler_angle[1] * Real(0.5));
		Real sin_pitch = glm::sin(euler_angle[1] * Real(0.5));
		Real cos_yaw = glm::cos(euler_angle[2] * Real(0.5));
		Real sin_yaw = glm::sin(euler_angle[2] * Real(0.5));

		w_ = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw;
		x_ = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw;
		y_ = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw;
		z_ = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw;
	}

	template <typename Real>
	DYN_FUNC Vector<Real, 3> Quat<Real>::getEulerAngle() const
	{
		Vector<Real, 3> euler_angle;
		euler_angle[0] = atan2(Real(2.0) * (w_ * z_ + x_ * y_), Real(1.0) - Real(2.0) * (z_ * z_ + x_ * x_));
		Real tmp = (Real(2.0) * (w_ * x_ - y_ * z_));
		if (tmp > 1.0)
			tmp = 1.0;
		if (tmp < -1.0)
			tmp = -1.0;
		euler_angle[1] = glm::asin(tmp);
		euler_angle[2] = atan2(Real(2.0) * (w_ * y_ + z_ * x_), Real(1.0) - Real(2.0) * (x_ * x_ + y_ * y_));
		return euler_angle;
	}

	template <typename Real>
	DYN_FUNC Real Quat<Real>::getAngle() const
	{
		return glm::acos(w_) * (Real)(2);
	}

	template <typename Real>
	DYN_FUNC Real Quat<Real>::getAngle(const Quat<Real>& quat) const
	{
		return glm::acos(dot(quat)) * (Real)(2);
	}

	template <typename Real>
	DYN_FUNC Real Quat<Real>::dot(const Quat<Real> & quat) const
	{
		return w_ * quat.w() + x_ * quat.x() + y_ * quat.y() + z_ * quat.z();
	}

	template <typename Real>
	DYN_FUNC Quat<Real> Quat<Real>::getConjugate() const
	{
		return Quat<Real>(x_, -y_, -z_, -w_);
	}

	template <typename Real>
	DYN_FUNC const Vector<Real, 3> Quat<Real>::rotate(const Vector<Real, 3> v) const
	{
		const Real vx = Real(2.0) * v[0];
		const Real vy = Real(2.0) * v[1];
		const Real vz = Real(2.0) * v[2];
		const Real w2 = w_ * w_ - (Real)0.5;
		const Real dot2 = (x_ * vx + y_ * vy + z_ * vz);
		return Vector<Real, 3>
			(
			(vx * w2 + (y_ * vz - z_ * vy) * w_ + x_ * dot2),
				(vy * w2 + (z_ * vx - x_ * vz) * w_ + y_ * dot2),
				(vz * w2 + (x_ * vy - y_ * vx) * w_ + z_ * dot2)
				);
	}

	template <typename Real>
	DYN_FUNC void Quat<Real>::rotateVector(Vector<Real, 3>& v)
	{
		Real xlen = v.norm();
		if (xlen == 0.0f) return;

		Quat p(0, v[0], v[1], v[2]);
		Quat qbar(x_, -y_, -z_, -w_);
		Quat qtmp;
		qtmp = (*this)*p;
		qtmp = qtmp * qbar;
		qtmp.normalize();
		v[0] = qtmp.y_; v[1] = qtmp.z_; v[2] = qtmp.w_;
		v.normalize();
		v *= xlen;
	}


	template <typename Real>
	DYN_FUNC SquareMatrix<Real, 3> Quat<Real>::get3x3Matrix() const
	{
		Real x = x_, y = y_, z = z_, w = w_;
		Real x2 = x + x, y2 = y + y, z2 = z + z;
		Real xx = x2 * x, yy = y2 * y, zz = z2 * z;
		Real xy = x2 * y, xz = x2 * z, xw = x2 * w;
		Real yz = y2 * z, yw = y2 * w, zw = z2 * w;
		return SquareMatrix<Real, 3>(Real(1) - yy - zz, xy - zw, xz + yw,
			xy + zw, Real(1) - xx - zz, yz - xw,
			xz - yw, yz + xw, Real(1) - xx - yy);

	}

	template <typename Real>
	DYN_FUNC SquareMatrix<Real, 4> Quat<Real>::get4x4Matrix() const
	{
		Real x = x_, y = y_, z = z_, w = w_;
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
			w_ = s * Real(0.5);
			if (s != 0.0)
				s = Real(0.5) / s;
			x_ = s * (matrix(1, 2) - matrix(2, 1));
			y_ = s * (matrix(2, 0) - matrix(0, 2));
			z_ = s * (matrix(0, 1) - matrix(1, 0));
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
			q[3] = s * (matrix(j, k) - matrix(k, j));
			q[j] = s * (matrix(i, j) - matrix(j, i));
			q[k] = s * (matrix(i, k) - matrix(k, i));
			x_ = q[0];
			y_ = q[1];
			z_ = q[2];
			w_ = q[3];
		}
	}

	template <typename Real>
	DYN_FUNC Quat<Real>::Quat(const SquareMatrix<Real, 4>& matrix)
	{
		Real tr = matrix(0, 0) + matrix(1, 1) + matrix(2, 2);
		if (tr > 0.0)
		{
			Real s = glm::sqrt(tr + Real(1.0));
			w_ = s * Real(0.5);
			if (s != 0.0)
				s = Real(0.5) / s;
			x_ = s * (matrix(1, 2) - matrix(2, 1));
			y_ = s * (matrix(2, 0) - matrix(0, 2));
			z_ = s * (matrix(0, 1) - matrix(1, 0));
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
			q[3] = s * (matrix(j, k) - matrix(k, j));
			q[j] = s * (matrix(i, j) - matrix(j, i));
			q[k] = s * (matrix(i, k) - matrix(k, i));
			x_ = q[0];
			y_ = q[1];
			z_ = q[2];
			w_ = q[3];
		}
	}

	template <typename Real>
	DYN_FUNC void Quat<Real>::toRotationAxis(Real& rot, Vector<Real, 3>& axis) const
	{
		rot = 2.0f * glm::acos(x_);
		if (rot == 0) {
			axis[0] = axis[1] = 0; axis[2] = 1;
			return;
		}
		axis[0] = y_;
		axis[1] = z_;
		axis[2] = w_;
		axis.normalize();
	}
}
