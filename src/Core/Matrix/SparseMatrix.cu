#include <cuda_runtime.h>
#include "SparseMatrix.h"
#include "Algorithm/Reduction.h"
#include "Algorithm/Arithmetic.h"
#include "Algorithm/Function2Pt.h"


namespace dyno
{
	template <typename VarType>
	void SparseMatrix<VarType>::Initialize(DArrayMap<VarType>& s_matrix,DArray<VarType>& s_b)	
	{
		my_A.assign(s_matrix);
		my_b.assign(s_b);
		my_x(s_b.size());
		my_x.reset();
	}

	template <typename VarType>
	void SparseMatrix<VarType>::clear()
	{
		my_A.clear();
		my_b.clear();
		my_x.clear();
	}

	//compute transposed(A)*a
	template <typename VarType>
	__global__ void transposedA_a(DArrayMap<VarType> matrix_a,DArray<VarType> a, DArray<VarType> A_a)
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
			A_a[tx] = sum;
		}
	}

	//compute A*a
	template <typename VarType>
	__global__ void A_a(DArrayMap<VarType> matrix_a, DArray<VarType> a, DArray<VarType> A_a)
	{
		int tx = blockIdx.x*blockDim.x + threadIdx.x;

		if (tx < a.size())
		{
			VarType sum = 0;
			Map<int, VarType> map = matrix_a[tx];
			if (map.size() > 0)
			{
				auto pair_v = map.begin();
				for (int k = 0; k < map.size(); k++)
				{
					int key = pair_v->first;
					sum += (pair_v->second)*a[key];
				}
			}
			A_a[tx] = sum;
		}
	}

	template <typename VarType>
	void SparseMatrix<VarType>::CGLS(int i_max,VarType threshold)
	{
		//x_0
		DArray<VarType> x_0(my_b.size());
		x_0.reset();

		int itor = 0;
		Arithmetic<VarType> arith;

		uint pDims = cudaGridSize(my_b.size(), BLOCK_SIZE);

		VarType delta_0 = 10, delta_new = 10, delta_old = 10;
		DArray<VarType> b_new, temp1, temp2, my_r, my_d, my_q;
		b_new.resize(my_b.size()); temp1.resize(my_b.size()); temp2.resize(my_b.size()); my_r.resize(my_b.size()); my_d.resize(my_b.size()); my_q.resize(my_b.size());
		b_new.reset(); temp1.reset(); temp2.reset(); my_r.reset(); my_d.reset(); my_q.reset();

		//compute b_new
		transposedA_a << <pDims, BLOCK_SIZE >> > (my_A, my_b, b_new);

		//compute r=b_new-transposed(A)*A*x_0
		A_a << <pDims, BLOCK_SIZE >> > (my_A, x_0, temp1);
		transposedA_a << <pDims, BLOCK_SIZE >> > (my_A, temp1, temp2);
		Function2Pt::subtract(my_r, b_new, temp2);

		my_d.assign(my_r);

		//compute delta
		delta_new = arith.Dot(my_r, my_r);
		delta_0 = delta_new;
		while ((itor<i_max)&&(delta_new>(threshold*threshold*delta_0)))
		{
			//compute alpha and x(i+1)
			A_a << <pDims, BLOCK_SIZE >> > (my_A, my_d, my_q);
			VarType alpha = delta_new / arith.Dot(my_q, my_q);
			Function2Pt::saxpy(my_x, my_d, my_x, alpha);

			//compute r
			if (itor % 50 == 0)
			{
				//compute r=b_new-transposed(A)*A*x
				temp1.reset(); temp2.reset();
				A_a << <pDims, BLOCK_SIZE >> > (my_A, my_x, temp1);
				transposedA_a << <pDims, BLOCK_SIZE >> > (my_A, temp1, temp2);
				Function2Pt::subtract(my_r, b_new, temp2);
			}
			else
			{
				temp1.reset();
				transposedA_a << <pDims, BLOCK_SIZE >> > (my_A, my_q, temp1);
				Function2Pt::saxpy(my_r, temp1, my_r, alpha);
			}

			delta_old = delta_new;
			delta_new = arith.Dot(my_r, my_r);
			VarType beta = delta_new / delta_old;
			Function2Pt::saxpy(my_d, my_r, my_d, beta);

			itor++;
		}
	}
}