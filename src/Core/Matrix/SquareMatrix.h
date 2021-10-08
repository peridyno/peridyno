#pragma once
#include "MatrixBase.h"

namespace dyno {

	template <typename T, int Dim>
	class SquareMatrix : public MatrixBase
	{
	public:
		SquareMatrix() {}
		~SquareMatrix() {}
		virtual unsigned int rows() const;
		virtual unsigned int cols() const;
	};

	template <typename T, int Dim>
	class Transform : public MatrixBase
	{
	public:
		Transform() {}
		~Transform() {}
		virtual unsigned int rows() const;
		virtual unsigned int cols() const;
	};

}  //end of namespace dyno

