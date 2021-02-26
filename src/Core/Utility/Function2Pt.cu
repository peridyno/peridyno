#include <cuda_runtime.h>
#include <vector_types.h>
#include "Utility.h"

namespace dyno
{
	namespace Function2Pt
	{
		template <typename T, typename Function>
		__global__ void KerTwoPointFunc(T *out, T* a1, T* a2, int num, Function func)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= num) return;

			out[pId] = func(a1[pId], a2[pId]);
		}

		template <typename T, typename Function>
		__global__ void KerTwoPointFunc(T *out, T* a2, int num, Function func)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= num) return;

			out[pId] = func(out[pId], a2[pId]);
		}

		template <typename T>
		__global__ void KerSaxpy(T *zArr, T* xArr, T* yArr, T alpha, int num)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= num) return;

			zArr[pId] = alpha * xArr[pId] + yArr[pId];
		}


		template <typename T>
		void plus(GArray<T>& zArr, GArray<T>& xArr, GArray<T>& yArr)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerTwoPointFunc << <pDim, BLOCK_SIZE >> > (zArr.begin(), xArr.begin(), yArr.begin(), zArr.size(), PlusFunc<T>());

		}

		template <typename T>
		void subtract(GArray<T>& zArr, GArray<T>& xArr, GArray<T>& yArr)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerTwoPointFunc <<<pDim, BLOCK_SIZE >>> (zArr.begin(), xArr.begin(), yArr.begin(), zArr.size(), MinusFunc<T>());
		}


		template <typename T>
		void multiply(GArray<T>& zArr, GArray<T>& xArr, GArray<T>& yArr)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerTwoPointFunc << <pDim, BLOCK_SIZE >> > (zArr.begin(), xArr.begin(), yArr.begin(), zArr.size(), MultiplyFunc<T>());

		}

		template <typename T>
		void divide(GArray<T>& zArr, GArray<T>& xArr, GArray<T>& yArr)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerTwoPointFunc << <pDim, BLOCK_SIZE >> > (zArr.begin(), xArr.begin(), yArr.begin(), zArr.size(), DivideFunc<T>());

		}


		template <typename T>
		void saxpy(GArray<T>& zArr, GArray<T>& xArr, GArray<T>& yArr, T alpha)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerSaxpy << <pDim, BLOCK_SIZE >> > (zArr.begin(), xArr.begin(), yArr.begin(), alpha, zArr.size());
		}

		template void plus(GArray<int>&, GArray<int>&, GArray<int>&);
		template void plus(GArray<float>&, GArray<float>&, GArray<float>&);
		template void plus(GArray<double>&, GArray<double>&, GArray<double>&);

		template void subtract(GArray<int>&, GArray<int>&, GArray<int>&);
		template void subtract(GArray<float>&, GArray<float>&, GArray<float>&);
		template void subtract(GArray<double>&, GArray<double>&, GArray<double>&);

		template void multiply(GArray<int>&, GArray<int>&, GArray<int>&);
		template void multiply(GArray<float>&, GArray<float>&, GArray<float>&);
		template void multiply(GArray<double>&, GArray<double>&, GArray<double>&);

		template void divide(GArray<int>&, GArray<int>&, GArray<int>&);
		template void divide(GArray<float>&, GArray<float>&, GArray<float>&);
		template void divide(GArray<double>&, GArray<double>&, GArray<double>&);

		template void saxpy(GArray<float>&, GArray<float>&, GArray<float>&, float);
		template void saxpy(GArray<double>&, GArray<double>&, GArray<double>&, double);
	}
}