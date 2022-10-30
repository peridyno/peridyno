#include "Reduction.h"
#include <cassert>
#include <cfloat>
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
	Reduction<T>::Reduction(uint num)
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
	Reduction<T>* Reduction<T>::Create(uint n)
	{
		return new Reduction<T>(n);
	}


	template<typename T>
	uint Reduction<T>::getAuxiliaryArraySize(uint n)
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
	__global__ void KerReduce(const T *pData, uint n, T *pAux, Function func, T val)
	{
		//extern __shared__ T sharedMem[];

		SharedMemory<T> smem;
		T* sharedMem = smem.getPointer();

		uint tid = threadIdx.x;
		uint id = blockIdx.x*blockDim.x + threadIdx.x;
		sharedMem[tid] = (id < n ? pData[id] : val);
		__syncthreads();
		if (blockSize >= 512) { if (tid < 256)sharedMem[tid] = func(sharedMem[tid], sharedMem[tid + 256]);  __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128)sharedMem[tid] = func(sharedMem[tid], sharedMem[tid + 128]);  __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) sharedMem[tid] = func(sharedMem[tid], sharedMem[tid + 64]);   __syncthreads(); }
		if (tid < 32)KerReduceWarp<T, blockSize>(sharedMem, tid, func);
		if (tid == 0)pAux[blockIdx.x] = sharedMem[0];
	}

	template<typename T, typename Function>
	T Reduce(T* pData, uint num, T* pAux, Function func, T v0)
	{
		uint n = num;
		uint sharedMemSize = REDUCTION_BLOCK * sizeof(T);
		uint blockNum = cudaGridSize(num, REDUCTION_BLOCK);
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
			cudaMemcpyAsync(&val, subAux, sizeof(T), cudaMemcpyDeviceToHost);
		else 
			cudaMemcpyAsync(&val, pData, sizeof(T), cudaMemcpyDeviceToHost);

		return val;
	}

	template<typename T>
	T Reduction<T>::accumulate(T* val, uint num)
	{
		if (num != m_num)
			allocAuxiliaryArray(num);

		return Reduce(val, num, m_aux, PlusFunc<T>(), (T)0);
	}

	template<typename T>
	T Reduction<T>::maximum(T* val, uint num)
	{
		if (num != m_num)
			allocAuxiliaryArray(num);

		return Reduce(val, num, m_aux, MaximumFunc<T>(), -(T)REAL_MAX);
	}

	template<typename T>
	T Reduction<T>::minimum(T* val, uint num)
	{
		if (num != m_num)
			allocAuxiliaryArray(num);

		return Reduce(val, num, m_aux, MinimumFunc<T>(), (T)REAL_MAX);
	}

	template<typename T>
	T Reduction<T>::average(T* val, uint num)
	{
		if (num != m_num)
			allocAuxiliaryArray(num);

		return Reduce(val, num, m_aux, PlusFunc<T>(), (T)0) / num;
	}

	template<typename T>
	void Reduction<T>::allocAuxiliaryArray(uint num)
	{
		if (m_aux != nullptr)
		{
			cudaFree(m_aux);
		}

		m_num = num;

		m_auxNum = getAuxiliaryArraySize(num);
		cudaMalloc((void**)&m_aux, m_auxNum * sizeof(T));
	}

	template class Reduction<int>;
	template class Reduction<float>;
	template class Reduction<double>;

	Reduction<Vec3f>::Reduction()
		: m_num(0)
		, m_aux(NULL)
	{

	}

	Reduction<Vec3f>::~Reduction()
	{
		if (m_aux != nullptr)
			cudaFree(m_aux);
	}

	__global__ void R_SetupComponent(float* comp, Vec3f* raw, size_t num, size_t comp_id)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= num) return;

		comp[tId] = raw[tId][comp_id];
	}

	Vec3f Reduction<Vec3f>::accumulate(Vec3f * val, uint num)
	{
		Vec3f ret;

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


	Vec3f Reduction<Vec3f>::maximum(Vec3f* val, uint num)
	{
		Vec3f ret;

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

	Vec3f Reduction<Vec3f>::minimum(Vec3f* val, uint num)
	{
		Vec3f ret;

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


	Vec3f Reduction<Vec3f>::average(Vec3f* val, uint num)
	{
		Vec3f ret;

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

	void Reduction<Vec3f>::allocAuxiliaryArray(uint num)
	{
		if (m_aux != nullptr)
		{
			cudaFree(m_aux);
		}

		m_num = num;
		cudaMalloc((void**)&m_aux, m_num * sizeof(float));
	}


	Reduction<Vec3d>::Reduction()
		: m_num(0)
		, m_aux(NULL)
	{

	}

	Reduction<Vec3d>::~Reduction()
	{
		if (m_aux != nullptr)
			cudaFree(m_aux);
	}

	__global__ void R_SetupComponent(double* comp, Vec3d* raw, size_t num, size_t comp_id)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= num) return;

		comp[tId] = raw[tId][comp_id];
	}

	Vec3d Reduction<Vec3d>::accumulate(Vec3d * val, uint num)
	{
		Vec3d ret;

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


	Vec3d Reduction<Vec3d>::maximum(Vec3d* val, uint num)
	{
		Vec3d ret;

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

	Vec3d Reduction<Vec3d>::minimum(Vec3d* val, uint num)
	{
		Vec3d ret;

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


	Vec3d Reduction<Vec3d>::average(Vec3d* val, uint num)
	{
		Vec3d ret;

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

	void Reduction<Vec3d>::allocAuxiliaryArray(uint num)
	{
		if (m_aux != nullptr)
		{
			cudaFree(m_aux);
		}

		m_num = num;
		cudaMalloc((void**)&m_aux, m_num * sizeof(double));
	}
}