#include <cuda_runtime.h>
#include "SMAlgorithm.h"
#include "Algorithm/Reduction.h"
#include "Algorithm/Arithmetic.h"
#include "Algorithm/Function2Pt.h"


namespace dyno
{
	//compute transposed(A)*a
	template <typename VarType>
	__global__ void transposedA_a(
		DArrayMap<VarType> matrix_a,
		DArray<VarType> a, 
		DArray<VarType> Aa)
	{
		int tx = blockIdx.x*blockDim.x + threadIdx.x;

		if (tx < a.size())
		{
			VarType sum = 0;
			for (int k = 0; k < a.size(); k++)
			{
				Map<int, VarType> map = matrix_a[k];
				if (map.size() > 0)
				{
					auto pair_v = map.find(tx);
					if (pair_v != nullptr)
						sum += (pair_v->second)*a[k];
				}
			}
			Aa[tx] = sum;
		}
	}

	template<typename VarType>
	void multiply_transposedSM_by_vector(DArrayMap<VarType>& matrix_a, DArray<VarType>& a, DArray<VarType>& Aa)
	{
		uint pDims = cudaGridSize(a.size(), BLOCK_SIZE);
		transposedA_a << <pDims, BLOCK_SIZE >> > (
			matrix_a,
			a, 
			Aa);
		cuSynchronize();
	}

	template void multiply_transposedSM_by_vector<float>(DArrayMap<float>& matrix_a, DArray<float>& a, DArray<float>& Aa);
	template void multiply_transposedSM_by_vector<double>(DArrayMap<double>& matrix_a, DArray<double>& a, DArray<double>& Aa);


	//compute A*a
	template <typename VarType>
	__global__ void A_a(
		DArrayMap<VarType> matrix_a,
		DArray<VarType> a,
		DArray<VarType> Aa)
	{
		int tx = blockIdx.x*blockDim.x + threadIdx.x;

		if (tx < a.size())
		{
			VarType sum = 0;
			Map<int, VarType> map = matrix_a[tx];

			if (map.size() > 0)
			{
				for (auto pair_v = map.begin(); pair_v != map.end(); ++pair_v)
				{
					int key = pair_v->first;
					sum += (pair_v->second)*a[key];
				}
			}
			Aa[tx] = sum;
		}
	}

	template<typename VarType>
	void multiply_SM_by_vector(DArrayMap<VarType>& matrix_a, DArray<VarType>& a, DArray<VarType>& Aa)
	{
		uint pDims = cudaGridSize(a.size(), BLOCK_SIZE);
		A_a << <pDims, BLOCK_SIZE >> > (
			matrix_a,
			a,
			Aa);
		cuSynchronize();
	}

	template void multiply_SM_by_vector<float>(DArrayMap<float>& matrix_a, DArray<float>& a, DArray<float>& Aa);
	template void multiply_SM_by_vector<double>(DArrayMap<double>& matrix_a, DArray<double>& a, DArray<double>& Aa);


	template <typename VarType>
	__global__ void count_TM(
		DArray<uint> count,
		DArrayMap<VarType> matrix)
	{
		int tx = blockIdx.x*blockDim.x + threadIdx.x;
		if (tx >= count.size()) return;

		Map<int, VarType>& map = matrix[tx];
		if (map.size() > 0)
		{
			for (auto pair_v = map.begin(); pair_v != map.end(); ++pair_v)
			{
				int key = pair_v->first;
				atomicAdd(&count[key], 1);
			}
		}
	}

	template<typename VarType>
	void count_transposedM(DArray<uint>& count, DArrayMap<VarType>& matrix)
	{
		cuExecute(count.size(),
			count_TM,
			count,
			matrix);
	}

	template void count_transposedM<float>(DArray<uint>& count, DArrayMap<float>& matrix);
	template void count_transposedM<double>(DArray<uint>& count, DArrayMap<double>& matrix);


	template <typename VarType>
	__global__ void compute_TM_unorder(
		DArrayMap<VarType> tmatrix,
		DArrayMap<VarType> matrix)
	{
		int tx = blockIdx.x*blockDim.x + threadIdx.x;
		if (tx >= matrix.size()) return;

		Map<int, VarType>& map = matrix[tx];
		if (map.size() > 0)
		{
			for (auto pair_v = map.begin(); pair_v != map.end(); ++pair_v)
			{
				int key = pair_v->first;
				tmatrix[key].atomicInsert(Pair<int, VarType>(tx, pair_v->second));
			}
		}
	}

	template <typename VarType>
	__global__ void compute_TM(
		DArrayMap<VarType> tmatrix,
		DArrayMap<VarType> matrix)
	{
		int tx = blockIdx.x*blockDim.x + threadIdx.x;
		if (tx >= matrix.size()) return;

		Map<int, VarType>& map = matrix[tx];
		if (map.size() > 0)
		{
			for (auto pair_v = map.begin(); pair_v != map.end(); ++pair_v)
			{
				tmatrix[tx].insert(*pair_v);
			}
		}
	}
	template<typename VarType>
	void compute_transposedM(DArrayMap<VarType>& tmatrix, DArray<uint>& count, DArrayMap<VarType>& matrix)
	{
		DArrayMap<VarType> unorder_tm;
		unorder_tm.resize(count);
		cuExecute(count.size(),
			compute_TM_unorder,
			unorder_tm,
			matrix);

		cuExecute(count.size(),
			compute_TM,
			tmatrix,
			unorder_tm);

		unorder_tm.clear();
	}

	template void compute_transposedM<float>(DArrayMap<float>& tmatrix, DArray<uint>& count, DArrayMap<float>& matrix);
	template void compute_transposedM<double>(DArrayMap<double>& tmatrix, DArray<uint>& count, DArrayMap<double>& matrix);
}