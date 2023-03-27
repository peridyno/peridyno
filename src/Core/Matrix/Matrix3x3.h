#pragma once
#include <glm/mat3x3.hpp>
#include "Vector.h"
#include "SquareMatrix.h"

namespace dyno {

	template <typename T, int Dim> class Vector;

	/*
	 * SquareMatrix<T,3> are defined for C++ fundamental integers types and floating-point types
	 * Elements are stored in a column-major order
	 */
	template <typename T>
	class SquareMatrix<T, 3>
	{
	public:
		typedef T VarType;

		DYN_FUNC SquareMatrix();
		DYN_FUNC explicit SquareMatrix(T);
		DYN_FUNC SquareMatrix(T x00, T x01, T x02, T x10, T x11, T x12, T x20, T x21, T x22);
		DYN_FUNC SquareMatrix(const Vector<T, 3> &row1, const Vector<T, 3> &row2, const Vector<T, 3> &row3);

		DYN_FUNC SquareMatrix(const SquareMatrix<T, 3>&);
		DYN_FUNC ~SquareMatrix();

		DYN_FUNC  static unsigned int rows() { return 3; }
		DYN_FUNC  static unsigned int cols() { return 3; }

		DYN_FUNC T& operator() (unsigned int i, unsigned int j);
		DYN_FUNC const T& operator() (unsigned int i, unsigned int j) const;

		DYN_FUNC const Vector<T, 3> row(unsigned int i) const;
		DYN_FUNC const Vector<T, 3> col(unsigned int i) const;

		DYN_FUNC void setRow(unsigned int i, const Vector<T, 3>& vec);
		DYN_FUNC void setCol(unsigned int j, const Vector<T, 3>& vec);

		DYN_FUNC const SquareMatrix<T, 3> operator+ (const SquareMatrix<T, 3> &) const;
		DYN_FUNC SquareMatrix<T, 3>& operator+= (const SquareMatrix<T, 3> &);
		DYN_FUNC const SquareMatrix<T, 3> operator- (const SquareMatrix<T, 3> &) const;
		DYN_FUNC SquareMatrix<T, 3>& operator-= (const SquareMatrix<T, 3> &);
		DYN_FUNC const SquareMatrix<T, 3> operator* (const SquareMatrix<T, 3> &) const;
		DYN_FUNC SquareMatrix<T, 3>& operator*= (const SquareMatrix<T, 3> &);
		DYN_FUNC const SquareMatrix<T, 3> operator/ (const SquareMatrix<T, 3> &) const;
		DYN_FUNC SquareMatrix<T, 3>& operator/= (const SquareMatrix<T, 3> &);

		DYN_FUNC SquareMatrix<T, 3>& operator= (const SquareMatrix<T, 3> &);

		DYN_FUNC bool operator== (const SquareMatrix<T, 3> &) const;
		DYN_FUNC bool operator!= (const SquareMatrix<T, 3> &) const;

		DYN_FUNC const SquareMatrix<T, 3> operator* (const T&) const;
		DYN_FUNC SquareMatrix<T, 3>& operator*= (const T&);
		DYN_FUNC const SquareMatrix<T, 3> operator/ (const T&) const;
		DYN_FUNC SquareMatrix<T, 3>& operator/= (const T&);

		DYN_FUNC const Vector<T, 3> operator* (const Vector<T, 3> &) const;

		DYN_FUNC const SquareMatrix<T, 3> operator- (void) const;

		DYN_FUNC const SquareMatrix<T, 3> transpose() const;
		DYN_FUNC const SquareMatrix<T, 3> inverse() const;

		DYN_FUNC T determinant() const;
		DYN_FUNC T trace() const;
		DYN_FUNC T doubleContraction(const SquareMatrix<T, 3> &) const;//double contraction
		DYN_FUNC T frobeniusNorm() const;
		DYN_FUNC T oneNorm() const;
		DYN_FUNC T infNorm() const;

		DYN_FUNC static const SquareMatrix<T, 3> identityMatrix();

		DYN_FUNC T* getDataPtr() { return &data_[0].x; }

	protected:
		Vector<T, 3> data_[3]; //default: zero matrix
	};

	//make * operator commutative
	template <typename S, typename T>
	DYN_FUNC  const SquareMatrix<T, 3> operator* (S scale, const SquareMatrix<T, 3> &mat)
	{
		return mat * scale;
	}

	template class SquareMatrix<float, 3>;
	template class SquareMatrix<double, 3>;
	//convenient typedefs
	typedef SquareMatrix<float, 3> Mat3f;
	typedef SquareMatrix<double, 3> Mat3d;

}  //end of namespace dyno

#include "Matrix3x3.inl"
