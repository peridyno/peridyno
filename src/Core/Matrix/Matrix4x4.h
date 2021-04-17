#pragma once
#include <glm/mat4x4.hpp>
#include "SquareMatrix.h"

namespace dyno {

	template <typename T, int Dim> class Vector;

	/*
	 * SquareMatrix<T,4> are defined for C++ fundamental integer and floating-point types
	 */
	template <typename T>
	class SquareMatrix<T, 4>
	{
	public:
		typedef T VarType;

		DYN_FUNC SquareMatrix();
		DYN_FUNC explicit SquareMatrix(T);
		DYN_FUNC SquareMatrix(T x00, T x01, T x02, T x03,
			T x10, T x11, T x12, T x13,
			T x20, T x21, T x22, T x23,
			T x30, T x31, T x32, T x33);
		DYN_FUNC SquareMatrix(const Vector<T, 4> &row1, const Vector<T, 4> &row2, const Vector<T, 4> &row3, const Vector<T, 4> &row4);

		DYN_FUNC SquareMatrix(const SquareMatrix<T, 4>&);
		DYN_FUNC ~SquareMatrix();

		DYN_FUNC  static unsigned int rows() { return 4; }
		DYN_FUNC  static unsigned int cols() { return 4; }

		DYN_FUNC T& operator() (unsigned int i, unsigned int j);
		DYN_FUNC const T& operator() (unsigned int i, unsigned int j) const;

		DYN_FUNC const Vector<T, 4> row(unsigned int i) const;
		DYN_FUNC const Vector<T, 4> col(unsigned int i) const;

		DYN_FUNC void setRow(unsigned int i, const Vector<T, 4>& vec);
		DYN_FUNC void setCol(unsigned int j, const Vector<T, 4>& vec);

		DYN_FUNC const SquareMatrix<T, 4> operator+ (const SquareMatrix<T, 4> &) const;
		DYN_FUNC SquareMatrix<T, 4>& operator+= (const SquareMatrix<T, 4> &);
		DYN_FUNC const SquareMatrix<T, 4> operator- (const SquareMatrix<T, 4> &) const;
		DYN_FUNC SquareMatrix<T, 4>& operator-= (const SquareMatrix<T, 4> &);
		DYN_FUNC const SquareMatrix<T, 4> operator* (const SquareMatrix<T, 4> &) const;
		DYN_FUNC SquareMatrix<T, 4>& operator*= (const SquareMatrix<T, 4> &);
		DYN_FUNC const SquareMatrix<T, 4> operator/ (const SquareMatrix<T, 4> &) const;
		DYN_FUNC SquareMatrix<T, 4>& operator/= (const SquareMatrix<T, 4> &);

		DYN_FUNC SquareMatrix<T, 4>& operator= (const SquareMatrix<T, 4> &);

		DYN_FUNC bool operator== (const SquareMatrix<T, 4> &) const;
		DYN_FUNC bool operator!= (const SquareMatrix<T, 4> &) const;

		DYN_FUNC const SquareMatrix<T, 4> operator* (const T&) const;
		DYN_FUNC SquareMatrix<T, 4>& operator*= (const T&);
		DYN_FUNC const SquareMatrix<T, 4> operator/ (const T&) const;
		DYN_FUNC SquareMatrix<T, 4>& operator/= (const T&);

		DYN_FUNC const Vector<T, 4> operator* (const Vector<T, 4> &) const;

		DYN_FUNC const SquareMatrix<T, 4> operator- (void) const;

		DYN_FUNC const SquareMatrix<T, 4> transpose() const;
		DYN_FUNC const SquareMatrix<T, 4> inverse() const;

		DYN_FUNC T determinant() const;
		DYN_FUNC T trace() const;
		DYN_FUNC T doubleContraction(const SquareMatrix<T, 4> &) const;//double contraction
		DYN_FUNC T frobeniusNorm() const;
		DYN_FUNC T oneNorm() const;
		DYN_FUNC T infNorm() const;

		DYN_FUNC static const SquareMatrix<T, 4> identityMatrix();

		DYN_FUNC T* getDataPtr() { return &data_[0].x; }

	protected:
		glm::tmat4x4<T> data_; //default: zero matrix
	};

	template class SquareMatrix<float, 4>;
	template class SquareMatrix<double, 4>;
	//convenient typedefs
	typedef SquareMatrix<float, 4> Mat4f;
	typedef SquareMatrix<double, 4> Mat4d;

}  //end of namespace dyno

#include "Matrix4x4.inl"
