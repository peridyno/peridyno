#include <cuda_runtime.h>
#include "SparseMatrix.h"
#include "Algorithm/Reduction.h"
#include "Algorithm/Arithmetic.h"
#include "Algorithm/Function2Pt.h"


namespace dyno
{
	template <typename VarType>
	SparseMatrix<VarType>::SparseMatrix(const SparseM s_matrix, const SparseV s_b)
	{
		A.assign(s_matrix);
		b.assign(s_b);
		x.assign(s_b);
		x.reset();
	}

	template <typename VarType>
	void SparseMatrix<VarType>::clear()
	{
		A.clear();
		b.clear();
		x.clear();
	}

	//compute transposed(A)*a
	template <typename SparseM, typename SparceV,typename VarType>
	__global__ void transposedA_a(SparseM matrix_a,SparceV a,SparceV A_a)
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
	template <typename SparseM, typename SparceV, typename VarType>
	__global__ void A_a(SparseM matrix_a, SparceV a, SparceV A_a)
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
		SparseV x_0(b.size());
		x_0.reset();

		int itor = 0;
		Arithmetic<VarType> arith;

		uint pDims = cudaGridSize(b.size(), BLOCK_SIZE);

		VarType delta_0 = 10, delta_new = 10, delta_old = 10;
		SparseV b_new, temp1, temp2, r, d, q;
		b_new.resize(b.size()); temp1.resize(b.size()); temp2.resize(b.size()); r.resize(b.size()); d.resize(b.size()); q.resize(b.size());
		b_new.reset(); temp1.reset(); temp2.reset(); r.reset(); d.reset(); q.reset();

		//compute b_new
		transposedA_a << <pDims, BLOCK_SIZE >> > (A, b, b_new);

		//compute r=b_new-transposed(A)*A*x_0
		A_a << <pDims, BLOCK_SIZE >> > (A, x_0, temp1);
		transposedA_a << <pDims, BLOCK_SIZE >> > (A, temp1, temp2);
		Function2Pt::subtract(r, b_new, temp2);

		d.assign(r);

		//compute delta
		delta_new = arith.Dot(r, r);
		delta_0 = delta_new;
		while ((itor<i_max)&&(delta_new>(threshold*threshold*delta_0)))
		{
			//compute alpha and x(i+1)
			A_a << <pDims, BLOCK_SIZE >> > (A, d, q);
			VarType alpha = delta_new / arith.Dot(q, q);
			Function2Pt::saxpy(x, d, x, alpha);

			//compute r
			if (itor % 50 == 0)
			{
				//compute r=b_new-transposed(A)*A*x
				temp1.reset(); temp2.reset();
				A_a << <pDims, BLOCK_SIZE >> > (A, x, temp1);
				transposedA_a << <pDims, BLOCK_SIZE >> > (A, temp1, temp2);
				Function2Pt::subtract(r, b_new, temp2);
			}
			else
			{
				temp1.reset();
				transposedA_a << <pDims, BLOCK_SIZE >> > (A, q, temp1);
				Function2Pt::saxpy(r, temp1, r, alpha);
			}

			delta_old = delta_new;
			delta_new = arith.Dot(r, r);
			VarType beta = delta_new / delta_old;
			Function2Pt::saxpy(d, r, d, beta);

			itor++;
		}
	}
}