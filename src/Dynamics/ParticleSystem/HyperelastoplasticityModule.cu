#include <cuda_runtime.h>
#include "HyperelastoplasticityModule.h"
#include "Framework/Node.h"
#include "Algorithm/MatrixFunc.h"
#include "Utility.h"
#include "ParticleSystem/Kernel.h"
#include <thrust/scan.h>
#include <thrust/reduce.h>
//#include "svd3_cuda2.h"

namespace dyno
{
	__device__ Real HPM_ConstantWeight(Real r, Real h)
	{
		return 1;
	}

	template<typename TDataType>
	HyperelastoplasticityModule<TDataType>::HyperelastoplasticityModule()
		: HyperelasticityModule_test<TDataType>()
	{
		this->attachField(&m_c, "c", "cohesion!", false);
		this->attachField(&m_phi, "phi", "friction angle!", false);

		m_c.setValue(0.001);
		m_phi.setValue(60.0 / 180.0);
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HPM_ComputeInvariants(
		GArray<Coord> stretching,
		GArray<bool> bYield,
		GArray<Real> yield_I1,
		GArray<Real> yield_J2,
		GArray<Real> arrI1,
		GArray<Coord> position,
		NeighborList<NPair> restShapes,
		Real A,
		Real B,
		Real mu,
		Real lambda)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		int size_i = restShapes.getNeighborSize(pId);

		Real total_weight = Real(0);
		Matrix matL_i(0);
		Matrix matK_i(0);

		Real maxDist = Real(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			maxDist = max(maxDist, r);
		}
		maxDist = maxDist < EPSILON ? Real(1) : maxDist;

		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > EPSILON)
			{
				Real weight = HPM_ConstantWeight(r, maxDist);

				Coord p = (position[j] - position[pId]) / maxDist;
				Coord q = (rest_pos_j - rest_pos_i) / maxDist;

				matL_i(0, 0) += p[0] * q[0] * weight; matL_i(0, 1) += p[0] * q[1] * weight; matL_i(0, 2) += p[0] * q[2] * weight;
				matL_i(1, 0) += p[1] * q[0] * weight; matL_i(1, 1) += p[1] * q[1] * weight; matL_i(1, 2) += p[1] * q[2] * weight;
				matL_i(2, 0) += p[2] * q[0] * weight; matL_i(2, 1) += p[2] * q[1] * weight; matL_i(2, 2) += p[2] * q[2] * weight;

				matK_i(0, 0) += q[0] * q[0] * weight; matK_i(0, 1) += q[0] * q[1] * weight; matK_i(0, 2) += q[0] * q[2] * weight;
				matK_i(1, 0) += q[1] * q[0] * weight; matK_i(1, 1) += q[1] * q[1] * weight; matK_i(1, 2) += q[1] * q[2] * weight;
				matK_i(2, 0) += q[2] * q[0] * weight; matK_i(2, 1) += q[2] * q[1] * weight; matK_i(2, 2) += q[2] * q[2] * weight;

				total_weight += weight;
			}
		}


		if (total_weight > EPSILON)
		{
			matL_i *= (1.0f / total_weight);
			matK_i *= (1.0f / total_weight);
		}

#ifdef DEBUG_INFO
		if (pId == 0)
		{
			Matrix mat_out = matK_i;
			printf("matK_i: \n %f %f %f \n %f %f %f \n %f %f %f \n\n",
				mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
				mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
				mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

			mat_out = matL_i;
			printf("matL_i: \n %f %f %f \n %f %f %f \n %f %f %f \n\n",
				mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
				mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
				mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

			mat_out = U * D*V.transpose();
			printf("matK polar: \n %f %f %f \n %f %f %f \n %f %f %f \n\n",
				mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
				mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
				mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

			printf("Horizon: %f; Det: %f \n", horizon, matK_i.determinant());
		}
#endif // DEBUG_INFO

		Matrix F_i = matL_i * matK_i.inverse();

		Real l0 = (F_i * Coord(1, 0, 0)).norm();
		Real l1 = (F_i * Coord(0, 1, 0)).norm();
		Real l2 = (F_i * Coord(0, 0, 1)).norm();

		Real slimit = Real(0.005);

		l0 = clamp(l0, Real(slimit), Real(1 / slimit));
		l1 = clamp(l1, Real(slimit), Real(1 / slimit));
		l2 = clamp(l2, Real(slimit), Real(1 / slimit));

		stretching[pId] = Coord(l0, l1, l2);

		if (pId == 0)
		{
			Matrix mat_out = F_i;
			printf("F_i: \n %f %f %f \n %f %f %f \n %f %f %f \n\n",
				mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
				mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
				mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

			printf("Yielding In HPM_ComputeInvariants: %f %f %f \n", l0, l1, l2);
		}

		Real I1_i = (l0 + l1 + l2) / 3;
		Real J2_i = 0;

		J2_i += (l0 - I1_i)*(l0 - I1_i);
		J2_i += (l1 - I1_i)*(l1 - I1_i);
		J2_i += (l2 - I1_i)*(l2 - I1_i);

		J2_i = sqrt(J2_i / 3);


		Real D1 = 1 - I1_i;		//positive for compression and negative for stretching

		Real yield_I1_i = 0.0f;
		Real yield_J2_i = 0.0f;

		Real s_J2 = J2_i*mu;
		Real s_D1 = D1*lambda;

		Real s_A = A;

		//Drucker-Prager yielding criterion
		if (s_J2 <= s_A + B*s_D1)
		{
			//bulk_stiffiness[i] = 10.0f;
			//invDeform[i] = Matrix::identityMatrix();
			yield_I1[pId] = Real(0);
			yield_J2[pId] = Real(0);
		}
		else
		{
			//bulk_stiffiness[i] = 0.0f;
			if (s_A + B*s_D1 > 0.0f)
			{
				yield_I1_i = 0.0f;

				yield_J2_i = (s_J2 - (s_A + B*s_D1)) / s_J2;
			}
			else
			{
				yield_I1_i = 1.0f;
				if (s_A + B*s_D1 < -EPSILON)
				{
					yield_I1_i = (s_A + B*s_D1) / (B*s_D1);
				}

				yield_J2_i = 1.0f;
			}

			yield_I1[pId] = yield_I1_i;
			yield_J2[pId] = yield_J2_i;
		}

		arrI1[pId] = I1_i;
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HPM_ApplyYielding(
		GArray<Coord> restPrincipleStretching,
		GArray<Real> yield_I1,
		GArray<Real> yield_J2,
		GArray<Real> arrI1,
		GArray<Coord> currentPrincipleStretching,
		GArray<Coord> position,
		NeighborList<NPair> restShape)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= position.size()) return;

		Coord rest_pos_i = restShape.getElement(i, 0).pos;
		Coord pos_i = position[i];

		Real yield_I1_i = yield_I1[i];
		Real yield_J2_i = yield_J2[i];
		Real I1_i = arrI1[i];

		Coord stretching = currentPrincipleStretching[i];

		Real l0 = stretching[0];
		Real l1 = stretching[1];
		Real l2 = stretching[2];

		Coord restStretching = restPrincipleStretching[i];

		Real l0_rest = restStretching[0];
		Real l1_rest = restStretching[1];
		Real l2_rest = restStretching[2];

		Real mul_i0 = Real(1) + (I1_i - 1)*yield_I1_i + (l0 - I1_i)*yield_J2_i;
		Real mul_i1 = Real(1) + (I1_i - 1)*yield_I1_i + (l1 - I1_i)*yield_J2_i;
		Real mul_i2 = Real(1) + (I1_i - 1)*yield_I1_i + (l2 - I1_i)*yield_J2_i;

		Real slimit = 0.8;
		Real l0_rest_new = clamp(l0_rest*mul_i0, slimit, 1 / slimit);
		Real l1_rest_new = clamp(l1_rest*mul_i1, slimit, 1 / slimit);
		Real l2_rest_new = clamp(l2_rest*mul_i2, slimit, 1 / slimit);

		restPrincipleStretching[i] = Coord(l0_rest_new, l1_rest_new, l2_rest_new);
	}


	//	int iter = 0;
	template<typename TDataType>
	bool HyperelastoplasticityModule<TDataType>::constrain()
	{
		this->solveElasticity();
		this->applyPlasticity();

		return true;
	}


/*	template<typename TDataType>
	void HyperelastoplasticityModule<TDataType>::solveElasticity()
	{
		Function1Pt::copy(this->m_position_old, this->inPosition()->getValue());

		this->computeInverseK();

		m_pbdModule->varIterationNumber()->setValue(1);

		int iter = 0;
		int total = this->getIterationNumber();
		while (iter < total)
		{
			this->enforceElasticity();
			if (m_incompressible.getValue() == true)
			{
				m_pbdModule->constrain();
			}
			
			iter++;
		}

		this->updateVelocity();
	}*/

	template<typename TDataType>
	void HyperelastoplasticityModule<TDataType>::applyPlasticity()
	{
		this->applyYielding();
	}


	template<typename TDataType>
	void HyperelastoplasticityModule<TDataType>::applyYielding()
	{
		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		Real A = computeA();
		Real B = computeB();

		HPM_ComputeInvariants<Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (
			m_yielding,
			m_bYield,
			m_yiled_I1,
			m_yield_J2,
			m_I1,
			this->inPosition()->getValue(),
			this->inRestShape()->getValue(),
			A,
			B,
			this->m_mu.getValue(),
			this->m_lambda.getValue());
		cuSynchronize();

		printf("P size: %d \n", this->inPosition()->getElementCount());
		printf("R size: %d \n", this->inRestShape()->getElementCount());

		// 
		HPM_ApplyYielding<Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (
			this->inPrincipleYielding()->getValue(),
			m_yiled_I1,
			m_yield_J2,
			m_I1,
			m_yielding,
			this->inPosition()->getValue(),
			this->inRestShape()->getValue());
		cuSynchronize();
	}

	template<typename TDataType>
	bool HyperelastoplasticityModule<TDataType>::initializeImpl()
	{
		m_yiled_I1.resize(this->inPosition()->getElementCount());
		m_yield_J2.resize(this->inPosition()->getElementCount());
		m_I1.resize(this->inPosition()->getElementCount());
		m_yielding.resize(this->inPosition()->getElementCount());

		return HyperelasticityModule_test<TDataType>::initializeImpl();
	}

	template<typename TDataType>
	void HyperelastoplasticityModule<TDataType>::begin()
	{
		int num = this->inPrincipleYielding()->getElementCount();

		if (num != m_yield_J2.size())
		{
			m_yiled_I1.resize(num);
			m_yield_J2.resize(num);
			m_I1.resize(num);

			m_yielding.resize(num);
		}

		HyperelasticityModule_test<TDataType>::begin();
	}


	template<typename TDataType>
	void HyperelastoplasticityModule<TDataType>::setCohesion(Real c)
	{
		m_c.setValue(c);
	}


	template<typename TDataType>
	void HyperelastoplasticityModule<TDataType>::setFrictionAngle(Real phi)
	{
		m_phi.setValue(phi/180);
	}
}