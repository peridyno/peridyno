#include "Reduction.h"
#include <cassert>
#include <cfloat>
#include <cuda_runtime.h>
#include "cuda_utilities.h"
#include "SharedMemory.h"
#include "Functional.h"

namespace dyno {

#define REDUCTION_BLOCK 128

	template<typename T>
	Reduction<T>::Reduction()
		: m_num(0)
		, m_aux(NULL)
	{

	}


	template<typename T>
	Reduction<T>::Reduction(unsigned num)
		: m_num(num)
		, m_aux(NULL)
	{
		allocAuxiliaryArray(m_num);
	}

	template<typename T>
	Reduction<T>::~Reduction()
	{
		if(m_aux != nullptr)
			cudaFree(m_aux);
	}

	template<typename T>
	Reduction<T>* Reduction<T>::Create(int n)
	{
		return new Reduction<T>(n);
	}


	template<typename T>
	int Reduction<T>::getAuxiliaryArraySize(int n)
	{
		return (n / REDUCTION_BLOCK + 1) + (n / (REDUCTION_BLOCK*REDUCTION_BLOCK) + REDUCTION_BLOCK);
	}

	/*!
	*	\brief	Reduction using maximum of float values in shared memory for a warp.
	*/
	template <typename T, 
			  unsigned blockSize,
			  typename Function>
	__device__ 	void KerReduceWarp(volatile T* pData, unsigned tid, Function func)
	{
		if (blockSize >= 64)pData[tid] = func(pData[tid], pData[tid + 32]);
		if (blockSize >= 32)pData[tid] = func(pData[tid], pData[tid + 16]);
		if (blockSize >= 16)pData[tid] = func(pData[tid], pData[tid + 8]);
		if (blockSize >= 8)pData[tid] = func(pData[tid], pData[tid + 4]);
		if (blockSize >= 4)pData[tid] = func(pData[tid], pData[tid + 2]);
		if (blockSize >= 2)pData[tid] = func(pData[tid], pData[tid + 1]);
	}

	/*!
	*	\brief	Accumulates the sum of n values of array pData[], 
	*	storing the result in the beginning of res[].
	*	(Many positions of res[] are used as blocks, storing the final result in res[0]).
	*/
	template <typename T, 
			  unsigned blockSize,
			  typename Function>
	__global__ void KerReduce(const T *pData, unsigned n, T *pAux, Function func, T val)
	{
		//extern __shared__ T sharedMem[];

		SharedMemory<T> smem;
		T* sharedMem = smem.getPointer();

		unsigned tid = threadIdx.x;
		unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
		sharedMem[tid] = (id < n ? pData[id] : val);
		__syncthreads();
		if (blockSize >= 512) { if (tid < 256)sharedMem[tid] = func(sharedMem[tid], sharedMem[tid + 256]);  __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128)sharedMem[tid] = func(sharedMem[tid], sharedMem[tid + 128]);  __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) sharedMem[tid] = func(sharedMem[tid], sharedMem[tid + 64]);   __syncthreads(); }
		if (tid < 32)KerReduceWarp<T, blockSize>(sharedMem, tid, func);
		if (tid == 0)pAux[blockIdx.x] = sharedMem[0];
	}

	template<typename T, typename Function>
	T Reduce(T* pData, unsigned num, T* pAux, Function func, T v0)
	{
		unsigned n = num;
		unsigned sharedMemSize = REDUCTION_BLOCK * sizeof(T);
		unsigned blockNum = cudaGridSize(num, REDUCTION_BLOCK);
		T* subData = pData;
		T* aux1 = pAux;
		T* aux2 = pAux + blockNum;
		T* subAux = aux1;
		while (n > 1) {
			KerReduce<T, REDUCTION_BLOCK, Function> << <blockNum, REDUCTION_BLOCK, sharedMemSize >> > (subData, n, subAux, func, v0);
			n = blockNum; 
			blockNum = cudaGridSize(n, REDUCTION_BLOCK);
			if (n > 1) {
				subData = subAux; subAux = (subData == aux1 ? aux2 : aux1);
			}
		}

		T val;
		if (num > 1)
			cudaMemcpy(&val, subAux, sizeof(T), cudaMemcpyDeviceToHost);
		else 
			cudaMemcpy(&val, pData, sizeof(T), cudaMemcpyDeviceToHost);

		return val;
	}

	template<typename T>
	T Reduction<T>::accumulate(T* val, int num)
	{
		if (num != m_num)
			allocAuxiliaryArray(num);

		return Reduce(val, num, m_aux, PlusFunc<T>(), (T)0);
	}

	template<typename T>
	T Reduction<T>::maximum(T* val, int num)
	{
		if (num != m_num)
			allocAuxiliaryArray(num);

		return Reduce(val, num, m_aux, MaximumFunc<T>(), (T)-FLT_MAX);
	}

	template<typename T>
	T Reduction<T>::minimum(T* val, int num)
	{
		if (num != m_num)
			allocAuxiliaryArray(num);

		return Reduce(val, num, m_aux, MinimumFunc<T>(), (T)FLT_MAX);
	}

	template<typename T>
	T Reduction<T>::average(T* val, int num)
	{
		if (num != m_num)
			allocAuxiliaryArray(num);

		return Reduce(val, num, m_aux, PlusFunc<T>(), (T)0) / num;
	}

	template<typename T>
	void Reduction<T>::allocAuxiliaryArray(int num)
	{
		if (m_aux != nullptr)
		{
			cudaFree(m_aux);
		}

		m_num = num;

		m_auxNum = getAuxiliaryArraySize(num);
		cudaMalloc((void**)&m_aux, m_auxNum * sizeof(T));
	}

	Reduction<Vector3f>::Reduction()
		: m_num(0)
		, m_aux(NULL)
	{

	}

	Reduction<Vector3f>::~Reduction()
	{
		if (m_aux != nullptr)
			cudaFree(m_aux);
	}

	__global__ void R_SetupComponent(float* comp, Vector3f* raw, int num, int comp_id)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= num) return;

		comp[tId] = raw[tId][comp_id];
	}

	Vector3f Reduction<Vector3f>::accumulate(Vector3f * val, int num)
	{
		Vector3f ret;

		if (num != m_num)
			allocAuxiliaryArray(num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			0);

		ret[0] = m_reduce_float.accumulate(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			1);

		ret[1] = m_reduce_float.accumulate(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			2);

		ret[2] = m_reduce_float.accumulate(m_aux, num);

		return ret;
	}


	Vector3f Reduction<Vector3f>::maximum(Vector3f* val, int num)
	{
		Vector3f ret;

		if (num != m_num)
			allocAuxiliaryArray(num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			0);

		ret[0] = m_reduce_float.maximum(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			1);

		ret[1] = m_reduce_float.maximum(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			2);

		ret[2] = m_reduce_float.maximum(m_aux, num);

		return ret;
	}

	Vector3f Reduction<Vector3f>::minimum(Vector3f* val, int num)
	{
		Vector3f ret;

		if (num != m_num)
			allocAuxiliaryArray(num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			0);

		ret[0] = m_reduce_float.minimum(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			1);

		ret[1] = m_reduce_float.minimum(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			2);

		ret[2] = m_reduce_float.minimum(m_aux, num);

		return ret;
	}


	Vector3f Reduction<Vector3f>::average(Vector3f* val, int num)
	{
		Vector3f ret;

		if (num != m_num)
			allocAuxiliaryArray(num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			0);

		ret[0] = m_reduce_float.average(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			1);

		ret[1] = m_reduce_float.average(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			2);

		ret[2] = m_reduce_float.average(m_aux, num);


		return ret;
	}

	void Reduction<Vector3f>::allocAuxiliaryArray(int num)
	{
		if (m_aux == nullptr)
		{
			cudaFree(m_aux);
		}

		m_num = num;
		cudaMalloc((void**)&m_aux, m_num * sizeof(float));
	}


	Reduction<Vector3d>::Reduction()
		: m_num(0)
		, m_aux(NULL)
	{

	}

	Reduction<Vector3d>::~Reduction()
	{
		if (m_aux != nullptr)
			cudaFree(m_aux);
	}

	__global__ void R_SetupComponent(double* comp, Vector3d* raw, int num, int comp_id)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= num) return;

		comp[tId] = raw[tId][comp_id];
	}

	Vector3d Reduction<Vector3d>::accumulate(Vector3d * val, int num)
	{
		Vector3d ret;

		if (num != m_num)
			allocAuxiliaryArray(num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			0);

		ret[0] = m_reduce_double.accumulate(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			1);

		ret[1] = m_reduce_double.accumulate(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			2);

		ret[2] = m_reduce_double.accumulate(m_aux, num);

		return ret;
	}


	Vector3d Reduction<Vector3d>::maximum(Vector3d* val, int num)
	{
		Vector3d ret;

		if (num != m_num)
			allocAuxiliaryArray(num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			0);

		ret[0] = m_reduce_double.maximum(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			1);

		ret[1] = m_reduce_double.maximum(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			2);

		ret[2] = m_reduce_double.maximum(m_aux, num);

		return ret;
	}

	Vector3d Reduction<Vector3d>::minimum(Vector3d* val, int num)
	{
		Vector3d ret;

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			0);

		ret[0] = m_reduce_double.minimum(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			1);

		ret[1] = m_reduce_double.minimum(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			2);

		ret[2] = m_reduce_double.minimum(m_aux, num);

		return ret;
	}


	Vector3d Reduction<Vector3d>::average(Vector3d* val, int num)
	{
		Vector3d ret;

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			0);

		ret[0] = m_reduce_double.average(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			1);

		ret[1] = m_reduce_double.average(m_aux, num);

		cuExecute(num,
			R_SetupComponent,
			m_aux,
			val,
			num,
			2);

		ret[2] = m_reduce_double.average(m_aux, num);


		return ret;
	}

	void Reduction<Vector3d>::allocAuxiliaryArray(int num)
	{
		if (m_aux == nullptr)
		{
			cudaFree(m_aux);
		}

		m_num = num;
		cudaMalloc((void**)&m_aux, m_num * sizeof(double));
	}
}