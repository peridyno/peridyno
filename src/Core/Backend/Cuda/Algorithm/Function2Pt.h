#pragma once
#include "Array/Array.h"
#include "Array/Array2D.h"
#include "Array/Array3D.h"
/*
*  This file implements two-point functions on device array types (DArray, DArray2D, DArray3D, etc.)
*/

namespace dyno
{
	namespace Function2Pt
	{
		// z = x + y;
		template <typename T>
		void plus(DArray<T>& zArr, DArray<T>& xArr, DArray<T>& yArr);

		// z = x - y;
		template <typename T>
		void subtract(DArray<T>& zArr, DArray<T>& xArr, DArray<T>& yArr);

		// z = x * y;
		template <typename T>
		void multiply(DArray<T>& zArr, DArray<T>& xArr, DArray<T>& yArr);

		// z = x / y;
		template <typename T>
		void divide(DArray<T>& zArr, DArray<T>& xArr, DArray<T>& yArr);

		// z = a * x + y;
		template <typename T>
		void saxpy(DArray<T>& zArr, DArray<T>& xArr, DArray<T>& yArr, T alpha);
	};
}
