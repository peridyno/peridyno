#pragma once
#include "Array/Array.h"
#include "Array/Array2D.h"
#include "Array/Array3D.h"
/*
*  This file implements two-point functions on device array types (DeviceArray, DeviceArray2D, DeviceArray3D, etc.)
*/

namespace dyno
{
	namespace Function2Pt
	{
		// z = x + y;
		template <typename T>
		void plus(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr);

		// z = x - y;
		template <typename T>
		void subtract(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr);

		// z = x * y;
		template <typename T>
		void multiply(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr);

		// z = x / y;
		template <typename T>
		void divide(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr);

		// z = a * x + y;
		template <typename T>
		void saxpy(DeviceArray<T>& zArr, DeviceArray<T>& xArr, DeviceArray<T>& yArr, T alpha);
	};
}
