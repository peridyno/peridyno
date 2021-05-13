#include <cuda_runtime.h>
#include "SparseMatrix.h"


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

	template <typename VarType>
	void SparseMatrix<VarType>::CGLS()
	{

	}

	// compute Ax;
	template <typename Real, typename Coord>
	__global__ void VC_ComputeAx
	(
		DArray<Real> residual,
		DArray<Real> pressure,
		DArray<Real> aiiSymArr,
		DArray<Real> alpha,
		DArray<Coord> position,
		DArray<Attribute> attribute,
		DArrayList<int> neighbors,
		Real smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		if (!attribute[pId].isDynamic()) return;

		Coord pos_i = position[pId];
		Real invAlpha_i = 1.0f / alpha[pId];

		atomicAdd(&residual[pId], aiiSymArr[pId] * pressure[pId]);
		Real con1 = 1.0f;// PARAMS.mass / PARAMS.restDensity / PARAMS.restDensity;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - position[j]).norm();

			if (r > EPSILON && attribute[j].isDynamic())
			{
				Real wrr_ij = kernWRR(r, smoothingLength);
				Real a_ij = -invAlpha_i * wrr_ij;
				//				residual += con1*a_ij*preArr[j];
				atomicAdd(&residual[pId], con1*a_ij*pressure[j]);
				atomicAdd(&residual[j], con1*a_ij*pressure[pId]);
			}
		}
	}






}