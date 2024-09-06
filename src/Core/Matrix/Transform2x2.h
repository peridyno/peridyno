#pragma once
#include "Quat.h"
#include "Matrix2x2.h"
namespace dyno {

	template <typename T, int Dim> class Vector;

	/*
	 * SquareMatrix<T,3> are defined for C++ fundamental integers types and floating-point types
	 */
	template <typename T>
	class Transform<T, 2>
	{
	public:
		typedef T VarType;

		DYN_FUNC Transform();
		DYN_FUNC Transform(const Vector<T, 2>& t, const T& angle, const Vector<T, 2>& s = Vector<T, 2>(1));
		DYN_FUNC Transform(const Vector<T, 2>& t, const SquareMatrix<T, 2>& m, const Vector<T, 2>& s = Vector<T, 2>(1));

		DYN_FUNC Transform(const Transform<T, 2>&);
		DYN_FUNC ~Transform();

		DYN_FUNC  static unsigned int rows() { return 2; }
		DYN_FUNC  static unsigned int cols() { return 2; }

		DYN_FUNC inline SquareMatrix<T, 2>& rotation() { return mRotation; }
		DYN_FUNC inline const SquareMatrix<T, 2> rotation() const { return mRotation; }

		DYN_FUNC inline Vector<T, 2>& translation() { return mTranslation; }
		DYN_FUNC inline const Vector<T, 2> translation() const { return mTranslation; }

		DYN_FUNC inline Vector<T, 2>& scale() { return mScale; }
		DYN_FUNC inline const Vector<T, 2> scale() const { return mScale; }

		DYN_FUNC const Vector<T, 2> operator* (const Vector<T, 2> &) const;

	protected:
		Vector<T, 2> mTranslation; //default: zero matrix
		Vector<T, 2> mScale;
		SquareMatrix<T, 2> mRotation;
	};

	template class Transform<float, 2>;
	template class Transform<double, 2>;
	//convenient typedefs
	typedef Transform<float, 2> Transform2f;
	typedef Transform<double, 2> Transform2d;
}  //end of namespace dyno

#include "Transform2x2.inl"
