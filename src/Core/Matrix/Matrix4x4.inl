#include <cmath>
#include <limits>

#include "Vector.h"

namespace dyno {

	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>::SquareMatrix()
		:SquareMatrix(0) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>::SquareMatrix(T value)
		: SquareMatrix(value, value, value, value,
			value, value, value, value,
			value, value, value, value,
			value, value, value, value) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>::SquareMatrix(T x00, T x01, T x02, T x03, T x10, T x11, T x12, T x13, T x20, T x21, T x22, T x23, T x30, T x31, T x32, T x33)
		:data_(x00, x10, x20, x30,
			x01, x11, x21, x31,
			x02, x12, x22, x32,
			x03, x13, x23, x33)
	{
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>::SquareMatrix(const Vector<T, 4> &row1, const Vector<T, 4> &row2, const Vector<T, 4> &row3, const Vector<T, 4> &row4)
		:data_(row1[0], row2[0], row3[0], row4[0],
			row1[1], row2[1], row3[1], row4[1],
			row1[2], row2[2], row3[2], row4[2],
			row1[3], row2[3], row3[3], row4[3])
	{
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>::SquareMatrix(const SquareMatrix<T, 4>& mat)
	{
		(*this)(0, 0) = mat(0, 0);	(*this)(0, 1) = mat(0, 1);	(*this)(0, 2) = mat(0, 2);	(*this)(0, 3) = mat(0, 3);
		(*this)(1, 0) = mat(1, 0);	(*this)(1, 1) = mat(1, 1);	(*this)(1, 2) = mat(1, 2);	(*this)(1, 3) = mat(1, 3);
		(*this)(2, 0) = mat(2, 0);	(*this)(2, 1) = mat(2, 1);	(*this)(2, 2) = mat(2, 2);	(*this)(2, 3) = mat(2, 3);
		(*this)(3, 0) = mat(3, 0);	(*this)(3, 1) = mat(3, 1);	(*this)(3, 2) = mat(3, 2);	(*this)(3, 3) = mat(3, 3);
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>::~SquareMatrix()
	{

	}

	template <typename T>
	DYN_FUNC T& SquareMatrix<T, 4>::operator() (unsigned int i, unsigned int j)
	{
		return const_cast<T &>(static_cast<const SquareMatrix<T, 4> &>(*this)(i, j));
	}

	template <typename T>
	DYN_FUNC const T& SquareMatrix<T, 4>::operator() (unsigned int i, unsigned int j) const
	{
		return data_[j][i];
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> SquareMatrix<T, 4>::row(unsigned int i) const
	{
		Vector<T, 4> result((*this)(i, 0), (*this)(i, 1), (*this)(i, 2), (*this)(i, 3));
		return result;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> SquareMatrix<T, 4>::col(unsigned int i) const
	{
		Vector<T, 4> result((*this)(0, i), (*this)(1, i), (*this)(2, i), (*this)(3, i));
		return result;
	}

	template <typename T>
	DYN_FUNC void SquareMatrix<T, 4>::setRow(unsigned int i, const Vector<T, 4>& vec)
	{
		data_[0][i] = vec[0];
		data_[1][i] = vec[1];
		data_[2][i] = vec[2];
		data_[3][i] = vec[3];
	}

	template <typename T>
	DYN_FUNC void SquareMatrix<T, 4>::setCol(unsigned int j, const Vector<T, 4>& vec)
	{
		data_[j][0] = vec[0];
		data_[j][1] = vec[1];
		data_[j][2] = vec[2];
		data_[j][3] = vec[3];
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 4> SquareMatrix<T, 4>::operator+ (const SquareMatrix<T, 4> &mat2) const
	{
		return SquareMatrix<T, 4>(*this) += mat2;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>& SquareMatrix<T, 4>::operator+= (const SquareMatrix<T, 4> &mat2)
	{
		data_ += mat2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 4> SquareMatrix<T, 4>::operator- (const SquareMatrix<T, 4> &mat2) const
	{
		return SquareMatrix<T, 4>(*this) -= mat2;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>& SquareMatrix<T, 4>::operator-= (const SquareMatrix<T, 4> &mat2)
	{
		data_ -= mat2.data_;
		return *this;
	}


	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>& SquareMatrix<T, 4>::operator=(const SquareMatrix<T, 4> &mat2)
	{
		data_ = mat2.data_;
		return *this;
	}


	template <typename T>
	DYN_FUNC bool SquareMatrix<T, 4>::operator== (const SquareMatrix<T, 4> &mat2) const
	{
		return data_ == mat2.data_;
	}

	template <typename T>
	DYN_FUNC bool SquareMatrix<T, 4>::operator!= (const SquareMatrix<T, 4> &mat2) const
	{
		return !((*this) == mat2);
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 4> SquareMatrix<T, 4>::operator* (const T& scale) const
	{
		return SquareMatrix<T, 4>(*this) *= scale;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>& SquareMatrix<T, 4>::operator*= (const T& scale)
	{
		data_ *= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> SquareMatrix<T, 4>::operator* (const Vector<T, 4> &vec) const
	{
		Vector<T, 4> result(0);
		for (unsigned int i = 0; i < 4; ++i)
			for (unsigned int j = 0; j < 4; ++j)
				result[i] += (*this)(i, j)*vec[j];
		return result;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 4> SquareMatrix<T, 4>::operator* (const SquareMatrix<T, 4> &mat2) const
	{
		return SquareMatrix<T, 4>(*this) *= mat2;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>& SquareMatrix<T, 4>::operator*= (const SquareMatrix<T, 4> &mat2)
	{
		data_ *= mat2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 4> SquareMatrix<T, 4>::operator/ (const SquareMatrix<T, 4> &mat2) const
	{
		return SquareMatrix<T, 4>(*this) *= mat2.inverse();
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>& SquareMatrix<T, 4>::operator/= (const SquareMatrix<T, 4> &mat2)
	{
		data_ *= glm::inverse(mat2.data_);
		return *this;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 4> SquareMatrix<T, 4>::operator/ (const T& scale) const
	{
		return SquareMatrix<T, 4>(*this) /= scale;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 4>& SquareMatrix<T, 4>::operator/= (const T& scale)
	{
		data_ /= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 4> SquareMatrix<T, 4>::operator- (void) const
	{
		SquareMatrix<T, 4> res;
		res.data_ = -data_;
		return res;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 4> SquareMatrix<T, 4>::transpose() const
	{
		SquareMatrix<T, 4> res;
		res.data_ = glm::transpose(data_);
		return res;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 4> SquareMatrix<T, 4>::inverse() const
	{
		SquareMatrix<T, 4> res;
		res.data_ = glm::inverse(data_);

		return res;
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 4>::determinant() const
	{
		return glm::determinant(data_);
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 4>::trace() const
	{
		return (*this)(0, 0) + (*this)(1, 1) + (*this)(2, 2) + (*this)(3, 3);
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 4>::doubleContraction(const SquareMatrix<T, 4> &mat2) const
	{
		T result = 0;
		for (unsigned int i = 0; i < 4; ++i)
			for (unsigned int j = 0; j < 4; ++j)
				result += (*this)(i, j)*mat2(i, j);
		return result;
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 4>::frobeniusNorm() const
	{
		T result = 0;
		for (unsigned int i = 0; i < 4; ++i)
			for (unsigned int j = 0; j < 4; ++j)
				result += (*this)(i, j)*(*this)(i, j);
		return glm::sqrt(result);
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 4>::oneNorm() const
	{
		const SquareMatrix<T, 4>& A = (*this);
		const T sum1 = fabs(A(0, 0)) + fabs(A(1, 0)) + fabs(A(2, 0)) + fabs(A(3, 0));
		const T sum2 = fabs(A(0, 1)) + fabs(A(1, 1)) + fabs(A(2, 1)) + fabs(A(3, 1));
		const T sum3 = fabs(A(0, 2)) + fabs(A(1, 2)) + fabs(A(2, 2)) + fabs(A(3, 2));
		const T sum4 = fabs(A(0, 3)) + fabs(A(1, 3)) + fabs(A(2, 3)) + fabs(A(3, 3));
		T maxSum = sum1;
		if (sum2 > maxSum)
			maxSum = sum2;
		if (sum3 > maxSum)
			maxSum = sum3;
		if (sum4 > maxSum)
			maxSum = sum4;
		return maxSum;
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 4>::infNorm() const
	{
		const SquareMatrix<T, 4>& A = (*this);
		const T sum1 = fabs(A(0, 0)) + fabs(A(0, 1)) + fabs(A(0, 2)) + fabs(A(0, 3));
		const T sum2 = fabs(A(1, 0)) + fabs(A(1, 1)) + fabs(A(1, 2)) + fabs(A(1, 3));
		const T sum3 = fabs(A(2, 0)) + fabs(A(2, 1)) + fabs(A(2, 2)) + fabs(A(2, 3));
		const T sum4 = fabs(A(3, 0)) + fabs(A(3, 1)) + fabs(A(3, 2)) + fabs(A(3, 3));
		T maxSum = sum1;
		if (sum2 > maxSum)
			maxSum = sum2;
		if (sum3 > maxSum)
			maxSum = sum3;
		if (sum4 > maxSum)
			maxSum = sum4;
		return maxSum;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 4> SquareMatrix<T, 4>::identityMatrix()
	{
		return SquareMatrix<T, 4>(1.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 1.0);
	}

	template <typename S, typename T>
	DYN_FUNC  const SquareMatrix<T, 4> operator* (S scale, const SquareMatrix<T, 4> &mat)
	{
		return mat * scale;
	}

}  //end of namespace dyno
