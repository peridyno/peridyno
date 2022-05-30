#pragma once
#include "Quat.h"
#include "Matrix3x3.h"
namespace dyno {

	template <typename T, int Dim> class Vector;

	/*
	 * SquareMatrix<T,3> are defined for C++ fundamental integers types and floating-point types
	 */
	template <typename T>
	class Transform<T, 3>
	{
	public:
		typedef T VarType;

		DYN_FUNC Transform();
		DYN_FUNC Transform(const Vector<T, 3>& t, const SquareMatrix<T, 3>& M, const Vector<T, 3>& s = Vector<T, 3>(1));

		DYN_FUNC Transform(const Transform<T, 3>&);
		DYN_FUNC ~Transform();

		DYN_FUNC  static unsigned int rows() { return 3; }
		DYN_FUNC  static unsigned int cols() { return 3; }

		DYN_FUNC inline SquareMatrix<T, 3>& rotation() { return mRotation; }
		DYN_FUNC inline const SquareMatrix<T, 3> rotation() const { return mRotation; }

		DYN_FUNC inline Vector<T, 3>& translation() { return mTranslation; }
		DYN_FUNC inline const Vector<T, 3> translation() const { return mTranslation; }

		DYN_FUNC inline Vector<T, 3>& scale() { return mScale; }
		DYN_FUNC inline const Vector<T, 3> scale() const { return mScale; }

		DYN_FUNC const Vector<T, 3> operator* (const Vector<T, 3> &) const;

	protected:
		Vector<T, 3> mTranslation; //default: zero matrix
		Vector<T, 3> mScale;
		SquareMatrix<T, 3> mRotation;
	};

	template class Transform<float, 3>;
	template class Transform<double, 3>;
	//convenient typedefs
	typedef Transform<float, 3> Transform3f;
	typedef Transform<double, 3> Transform3d;
}  //end of namespace dyno

#include "Transform3x3.inl"
