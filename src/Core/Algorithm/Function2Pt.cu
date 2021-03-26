#include <cuda_runtime.h>
#include <vector_types.h>
#include "Functional.h"
#include "Function2Pt.h"

namespace dyno
{
	namespace Function2Pt
	{
		template <typename T, typename Function>
		__global__ void KerTwoPointFunc(T *out, T* a1, T* a2, size_t num, Function func)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= num) return;

			out[pId] = func(a1[pId], a2[pId]);
		}

		template <typename T, typename Function>
		__global__ void KerTwoPointFunc(T *out, T* a2, size_t num, Function func)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= num) return;

			out[pId] = func(out[pId], a2[pId]);
		}

		template <typename T>
		__global__ void KerSaxpy(T *zArr, T* xArr, T* yArr, T alpha, size_t num)
		{
			int pId = threadIdx.x + (blockIdx.x * blockDim.x);
			if (pId >= num) return;

			zArr[pId] = alpha * xArr[pId] + yArr[pId];
		}


		template <typename T>
		void plus(DArray<T>& zArr, DArray<T>& xArr, DArray<T>& yArr)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerTwoPointFunc << <pDim, BLOCK_SIZE >> > (zArr.begin(), xArr.begin(), yArr.begin(), zArr.size(), PlusFunc<T>());

		}

		template <typename T>
		void subtract(DArray<T>& zArr, DArray<T>& xArr, DArray<T>& yArr)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerTwoPointFunc <<<pDim, BLOCK_SIZE >>> (zArr.begin(), xArr.begin(), yArr.begin(), zArr.size(), MinusFunc<T>());
		}


		template <typename T>
		void multiply(DArray<T>& zArr, DArray<T>& xArr, DArray<T>& yArr)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerTwoPointFunc << <pDim, BLOCK_SIZE >> > (zArr.begin(), xArr.begin(), yArr.begin(), zArr.size(), MultiplyFunc<T>());

		}

		template <typename T>
		void divide(DArray<T>& zArr, DArray<T>& xArr, DArray<T>& yArr)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerTwoPointFunc << <pDim, BLOCK_SIZE >> > (zArr.begin(), xArr.begin(), yArr.begin(), zArr.size(), DivideFunc<T>());

		}


		template <typename T>
		void saxpy(DArray<T>& zArr, DArray<T>& xArr, DArray<T>& yArr, T alpha)
		{
			assert(zArr.size() == xArr.size() && zArr.size() == yArr.size());
			unsigned pDim = cudaGridSize(zArr.size(), BLOCK_SIZE);
			KerSaxpy << <pDim, BLOCK_SIZE >> > (zArr.begin(), xArr.begin(), yArr.begin(), alpha, zArr.size());
		}

		template void plus(DArray<int>&, DArray<int>&, DArray<int>&);
		template void plus(DArray<float>&, DArray<float>&, DArray<float>&);
		template void plus(DArray<double>&, DArray<double>&, DArray<double>&);

		template void subtract(DArray<int>&, DArray<int>&, DArray<int>&);
		template void subtract(DArray<float>&, DArray<float>&, DArray<float>&);
		template void subtract(DArray<double>&, DArray<double>&, DArray<double>&);

		template void multiply(DArray<int>&, DArray<int>&, DArray<int>&);
		template void multiply(DArray<float>&, DArray<float>&, DArray<float>&);
		template void multiply(DArray<double>&, DArray<double>&, DArray<double>&);

		template void divide(DArray<int>&, DArray<int>&, DArray<int>&);
		template void divide(DArray<float>&, DArray<float>&, DArray<float>&);
		template void divide(DArray<double>&, DArray<double>&, DArray<double>&);

		template void saxpy(DArray<float>&, DArray<float>&, DArray<float>&, float);
		template void saxpy(DArray<double>&, DArray<double>&, DArray<double>&, double);
	}
}