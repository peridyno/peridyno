#pragma once
#include "Array/Array.h"
#include "Array/Array2D.h"
#include "Array/Array3D.h"
/*
*  This file implements two-point functions on device array types (GArray, GArray2D, GArray3D, etc.)
*/

namespace dyno
{
	namespace Function2Pt
	{
		// z = x + y;
		template <typename T>
		void plus(GArray<T>& zArr, GArray<T>& xArr, GArray<T>& yArr);

		// z = x - y;
		template <typename T>
		void subtract(GArray<T>& zArr, GArray<T>& xArr, GArray<T>& yArr);

		// z = x * y;
		template <typename T>
		void multiply(GArray<T>& zArr, GArray<T>& xArr, GArray<T>& yArr);

		// z = x / y;
		template <typename T>
		void divide(GArray<T>& zArr, GArray<T>& xArr, GArray<T>& yArr);

		// z = a * x + y;
		template <typename T>
		void saxpy(GArray<T>& zArr, GArray<T>& xArr, GArray<T>& yArr, T alpha);
	};
}
