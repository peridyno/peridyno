#pragma once
#include <glm/mat2x2.hpp>
#include "SquareMatrix.h"

namespace dyno {

	template <typename T, int Dim> class Vector;

	template <typename T>
	class SquareMatrix<T, 2>
	{
	public:
		typedef T VarType;

		DYN_FUNC SquareMatrix();
		DYN_FUNC explicit SquareMatrix(T);
		DYN_FUNC SquareMatrix(T x00, T x01, T x10, T x11);
		DYN_FUNC SquareMatrix(const Vector<T, 2> &row1, const Vector<T, 2> &row2);

		DYN_FUNC SquareMatrix(const SquareMatrix<T, 2> &);
		DYN_FUNC ~SquareMatrix();

		DYN_FUNC static unsigned int rows() { return 2; }
		DYN_FUNC static unsigned int cols() { return 2; }

		DYN_FUNC T& operator() (unsigned int i, unsigned int j);
		DYN_FUNC const T& operator() (unsigned int i, unsigned int j) const;

		DYN_FUNC const Vector<T, 2> row(unsigned int i) const;
		DYN_FUNC const Vector<T, 2> col(unsigned int i) const;

		DYN_FUNC void setRow(unsigned int i, const Vector<T, 2>& vec);
		DYN_FUNC void setCol(unsigned int j, const Vector<T, 2>& vec);

		DYN_FUNC const SquareMatrix<T, 2> operator+ (const SquareMatrix<T, 2> &) const;
		DYN_FUNC SquareMatrix<T, 2>& operator+= (const SquareMatrix<T, 2> &);
		DYN_FUNC const SquareMatrix<T, 2> operator- (const SquareMatrix<T, 2> &) const;
		DYN_FUNC SquareMatrix<T, 2>& operator-= (const SquareMatrix<T, 2> &);
		DYN_FUNC const SquareMatrix<T, 2> operator* (const SquareMatrix<T, 2> &) const;
		DYN_FUNC SquareMatrix<T, 2>& operator*= (const SquareMatrix<T, 2> &);
		DYN_FUNC const SquareMatrix<T, 2> operator/ (const SquareMatrix<T, 2> &) const;
		DYN_FUNC SquareMatrix<T, 2>& operator/= (const SquareMatrix<T, 2> &);

		DYN_FUNC SquareMatrix<T, 2>& operator= (const SquareMatrix<T, 2> &);

		DYN_FUNC bool operator== (const SquareMatrix<T, 2> &) const;
		DYN_FUNC bool operator!= (const SquareMatrix<T, 2> &) const;

		DYN_FUNC const SquareMatrix<T, 2> operator* (const T&) const;
		DYN_FUNC SquareMatrix<T, 2>& operator*= (const T&);
		DYN_FUNC const SquareMatrix<T, 2> operator/ (const T&) const;
		DYN_FUNC SquareMatrix<T, 2>& operator/= (const T&);

		DYN_FUNC const Vector<T, 2> operator* (const Vector<T, 2> &) const;

		DYN_FUNC const SquareMatrix<T, 2> operator- (void) const;

		DYN_FUNC const SquareMatrix<T, 2> transpose() const;
		DYN_FUNC const SquareMatrix<T, 2> inverse() const;

		DYN_FUNC T determinant() const;
		DYN_FUNC T trace() const;
		DYN_FUNC T doubleContraction(const SquareMatrix<T, 2> &) const;//double contraction
		DYN_FUNC T frobeniusNorm() const;
		DYN_FUNC T oneNorm() const;
		DYN_FUNC T infNorm() const;

		DYN_FUNC static const SquareMatrix<T, 2> identityMatrix();

		DYN_FUNC T* getDataPtr() { return &data_[0].x; }

	protected:
		glm::tmat2x2<T> data_; //default: zero matrix
	};

	//make * operator commutative
	template <typename S, typename T>
	DYN_FUNC const SquareMatrix<T, 2> operator* (S scale, const SquareMatrix<T, 2> &mat)
	{
		return mat * scale;
	}

	template class SquareMatrix<float, 2>;
	template class SquareMatrix<double, 2>;
	//convenient typedefs
	typedef SquareMatrix<float, 2> Mat2f;
	typedef SquareMatrix<double, 2> Mat2d;

}  //end of namespace dyno

#include "Matrix2x2.inl"
