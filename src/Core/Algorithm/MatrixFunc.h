#pragma once
#include "Platform.h"
#include "Matrix.h"

namespace dyno
{
	template<typename Real, int Dim>
	DYN_FUNC void polarDecomposition(const SquareMatrix<Real, Dim> &A, SquareMatrix<Real, Dim> &R, SquareMatrix<Real, Dim> &U, SquareMatrix<Real, Dim> &D, SquareMatrix<Real, Dim> &V);

	template<typename Real, int Dim>
	DYN_FUNC void polarDecomposition(const SquareMatrix<Real, Dim> &A, SquareMatrix<Real, Dim> &R, SquareMatrix<Real, Dim> &U, SquareMatrix<Real, Dim> &D);

	template<typename Real, int Dim>
	DYN_FUNC void polarDecomposition(const SquareMatrix<Real, Dim> &M, SquareMatrix<Real, Dim> &R, Real tolerance);
}

#include "MatrixFunc.inl"