#pragma once
#include "../Vector.h"
#include "../Matrix.h"


namespace dyno 
{
	/*
	 * Quaternion is defined for float and double.
	 */
	template <typename Real>
	class Quaternion
	{
	public:
		/* Constructors */
		DYN_FUNC Quaternion();
		DYN_FUNC Quaternion(Real x, Real y, Real z, Real w);
		DYN_FUNC Quaternion(const Vector<Real, 3> &unit_axis, Real angle_rad);  //init from the rotation axis and angle(in radian)
		DYN_FUNC Quaternion(Real angle_rad, const Vector<Real, 3> &unit_axis);
		DYN_FUNC explicit Quaternion(const Real *);
		DYN_FUNC Quaternion(const Quaternion<Real> &);
		DYN_FUNC explicit Quaternion(const SquareMatrix<Real, 3> &);   //init from a 3x3matrix
		DYN_FUNC explicit Quaternion(const SquareMatrix<Real, 4> &);    //init from a 4x4matrix
		DYN_FUNC explicit Quaternion(const Vector<Real, 3>&);         //init form roll pitch yaw/ Euler angle;

		/* Assignment operators */
		DYN_FUNC Quaternion<Real> &operator = (const Quaternion<Real> &);
		DYN_FUNC Quaternion<Real> &operator += (const Quaternion<Real> &);
		DYN_FUNC Quaternion<Real> &operator -= (const Quaternion<Real> &);

		/* Get and Set functions */
		DYN_FUNC inline Real x() const { return x_; }
		DYN_FUNC inline Real y() const { return y_; }
		DYN_FUNC inline Real z() const { return z_; }
		DYN_FUNC inline Real w() const { return w_; }

		DYN_FUNC inline void setX(const Real& x) { x_ = x; }
		DYN_FUNC inline void setY(const Real& y) { y_ = y; }
		DYN_FUNC inline void setZ(const Real& z) { z_ = z; }
		DYN_FUNC inline void setW(const Real& w) { w_ = w; }

		//rotate
		DYN_FUNC const Vector<Real, 3> rotate(const Vector<Real, 3>) const;    // rotates passed vec by this.
		DYN_FUNC void rotateVector(Vector<Real, 3>& v);
		DYN_FUNC void toRotationAxis(Real &rot, Vector<Real, 3> &axis) const;

		/* Special functions */
		DYN_FUNC Real norm();
		DYN_FUNC Quaternion<Real>& normalize();

		DYN_FUNC void set(const Vector<Real, 3>&, Real);
		DYN_FUNC void set(Real, const Vector<Real, 3>&);
		DYN_FUNC void set(const Vector<Real, 3>&);                              //set from a euler angle.

		DYN_FUNC Real getAngle() const;                                         // return the angle between this quat and the identity quaternion.
		DYN_FUNC Real getAngle(const Quaternion<Real>&) const;                // return the angle between this and the argument
		DYN_FUNC Quaternion<Real> getConjugate() const;                         // return the conjugate

		DYN_FUNC SquareMatrix<Real, 3> get3x3Matrix() const;                    //return 3x3matrix format
		DYN_FUNC SquareMatrix<Real, 4> get4x4Matrix() const;                    //return 4x4matrix with a identity transform.
		DYN_FUNC Vector<Real, 3> getEulerAngle() const;

		DYN_FUNC Quaternion<Real> multiply_q(const Quaternion<Real>&);

		/* Operator overloading */
		DYN_FUNC Quaternion<Real> operator - (const Quaternion<Real>&) const;
		DYN_FUNC Quaternion<Real> operator - (void) const;
		DYN_FUNC Quaternion<Real> operator + (const Quaternion<Real>&) const;
		DYN_FUNC Quaternion<Real> operator * (const Quaternion<Real>&) const;
		DYN_FUNC Quaternion<Real> operator * (const Real&) const;
		DYN_FUNC Quaternion<Real> operator / (const Real&) const;
		DYN_FUNC bool operator == (const Quaternion<Real>&) const;
		DYN_FUNC bool operator != (const Quaternion<Real>&) const;
		DYN_FUNC Real& operator[] (unsigned int);
		DYN_FUNC const Real& operator[] (unsigned int) const;
		DYN_FUNC Real dot(const Quaternion<Real> &) const;

		DYN_FUNC static inline Quaternion<Real> Identity() { return Quaternion<Real>(0, 0, 0, 1); }

	public:
		Real x_, y_, z_, w_;
	};

	//make * operator commutative
	template <typename S, typename T>
	DYN_FUNC inline Quaternion<T> operator *(S scale, const Quaternion<T> &quad)
	{
		return quad * scale;
	}

	template class Quaternion<float>;
	template class Quaternion<double>;
	//convenient typedefs
	typedef Quaternion<float> Quaternionf;
	typedef Quaternion<double> Quaterniond;

}//end of namespace dyno

#include "quaternion.inl"