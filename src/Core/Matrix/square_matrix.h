#pragma once
#include "matrix_base.h"

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

}  //end of namespace dyno

