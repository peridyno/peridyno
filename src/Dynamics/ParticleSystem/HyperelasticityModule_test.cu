#include "HyperelasticityModule_test.h"
#include "Utility.h"
#include "Utility/CudaRand.h"
#include "Framework/Node.h"
#include "Algorithm/MatrixFunc.h"
#include "Kernel.h"

#include "Framework/Log.h"
#include "Utility/Function1Pt.h"

#include "Hyperelasticity_computation_helper.cu"

//#define DEBUG_INFO

namespace dyno
{
	__device__ Real constantWeight(Real r, Real h)
	{
 		Real d = h / r;
 		return d*d;
//		return Real(0.25);
	}

	template<typename Real, typename Matrix>
	__device__ HyperelasticityModel<Real, Matrix>* getElasticityModel(EnergyType type)
	{
		switch (type)
		{
		case StVK:
			return new StVKModel<Real, Matrix>();
		case NeoHooekean:
			return new NeoHookeanModel<Real, Matrix>();
		case Polynomial:
			return new PolynomialModel<Real, Matrix, 1>();
		case Xuetal:
			return new XuModel<Real, Matrix>();
		default:
			break;
		}
	}

	IMPLEMENT_CLASS_1(HyperelasticityModule_test, TDataType)

	template<typename TDataType>
	HyperelasticityModule_test<TDataType>::HyperelasticityModule_test()
		: ElasticityModule<TDataType>()
		, m_energyType(NeoHooekean)
	{
	}

	template <typename Coord>
	__global__ void test_HM_UpdatePosition(
		GArray<Coord> position,
		GArray<Coord> velocity,
		GArray<Coord> y_next,
		GArray<Coord> position_old,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

#ifdef DEBUG_INFO
		if (pId == 0)
		{
			printf("UpdatePosition %d: %f %f %f \n", pId, y_next[pId][0], y_next[pId][1], y_next[pId][2]);
		}
#endif // DEBUG_INFO


		position[pId] = y_next[pId];
		velocity[pId] += (position[pId] - position_old[pId]) / dt;
	}

	//�������ֵ
	template <typename Real, typename Coord>
	__global__ void HM_Blend(
		GArray<Real> blend,
		GArray<Coord> eigens)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= blend.size()) return;

		Coord eigen_i = eigens[pId];
		Real l = min(eigen_i[0], min(eigen_i[1], eigen_i[2]));

		Real l_max = 0.8;
		Real l_min = 0.2;

		Real value = (l_max - l) / (l_max - l_min);
		blend[pId] = 1;// clamp(value, Real(0), Real(1));
	}

	template <typename Real, typename Matrix>
	__global__ void HM_ComputeFirstPiolaKirchhoff(
		GArray<Matrix> stressTensor,
		GArray<Matrix> F,
		GArray<Matrix> inverseF,
		Real mu,
		Real lambda)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= inverseF.size()) return;

		Matrix F_i = F[pId];

		// find strain tensor E = 1/2(F^T * F - I)
		Matrix E = 0.5*(F_i.transpose() * F_i - Matrix::identityMatrix());

		// find first Piola-Kirchhoff matix; StVK material
		stressTensor[pId] = F_i * (2 * lambda * E);
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_JacobiStepExplicit(
		GArray<Coord> velocity,
		GArray<Coord> y_new,
		GArray<Coord> y_old,
		GArray<Coord> source,
		GArray<Matrix> stressTensor,
		GArray<Matrix> invK,
		GArray<Matrix> invL,
		NeighborList<NPair> restShapes,
		Real horizon,
		Real mass,
		Real volume,
		Real mu,
		Real lambda,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_old.size()) return;

		const Real scale = volume*volume;

		int size_i = restShapes.getNeighborSize(pId);

		Coord y_i = y_old[pId];
		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		//Matrix PK_i = stressTensor[pId] * invK[pId];
		Matrix PK_i = stressTensor[pId] * invK[pId];

		Matrix mat_i(0);
		Coord source_i = source[pId];
		Coord force_i(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord y_j = y_old[j];
			Real r = (np_j.pos - rest_pos_i).norm();

			if (r > EPSILON)
			{
				Real weight = constantWeight(r, horizon);

				//Matrix PK_j = stressTensor[j] * invK[j];
				Matrix PK_j = stressTensor[j] * invK[j];

				Matrix PK_ij = scale * weight * (PK_i + PK_j);

				//force_i += PK_ij * (np_j.pos - rest_pos_i);
				force_i += PK_ij * (y_old[j] - y_old[pId]);
			}
		}


		velocity[pId] += force_i * dt / mass;

		if (pId == 0)
		{
			printf("force: %f %f %f \n", force_i[0], force_i[1], force_i[2]);

			printf("PK: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
				PK_i(0, 0), PK_i(0, 1), PK_i(0, 2),
				PK_i(1, 0), PK_i(1, 1), PK_i(1, 2),
				PK_i(2, 0), PK_i(2, 1), PK_i(2, 2));


			Matrix K_i = invK[pId];
			printf("invK: \n %f %f %f \n %f %f %f \n %f %f %f \n	\n",
				K_i(0, 0), K_i(0, 1), K_i(0, 2),
				K_i(1, 0), K_i(1, 1), K_i(1, 2),
				K_i(2, 0), K_i(2, 1), K_i(2, 2));
		}

	}

	template <typename Real>
	__device__ Real HM_Interpolant(Real lambda)
	{
		Real l_max = 0.15;
		Real l_min = 0.05;
		return clamp((l_max - lambda) / (l_max - l_min), Real(0), Real(1));
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_JacobiLinear(
		GArray<Coord> source,
		GArray<Matrix> A,
		GArray<Coord> y_pre,
		GArray<Matrix> F,
		NeighborList<NPair> restShapes,
		GArray<Real> volume,
		Real horizon,
		Real dt,
		GArray<Real> fraction)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= source.size()) return;

		Real frac_i = 1 - fraction[pId];

		Matrix F_i = F[pId];
		if ((F_i.determinant()) < -0.001f)
		{
			F_i = Matrix::identityMatrix();
		}

		Real V_i = volume[pId];

		Coord y_pre_i = y_pre[pId];
		Coord y_rest_i = restShapes.getElement(pId, 0).pos;

		Matrix mat_i(0);
		Coord source_i(0);

		int size_i = restShapes.getNeighborSize(pId);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord y_pre_j = y_pre[j];
			Real r = (np_j.pos - y_rest_i).norm();

			if (r > EPSILON)
			{
				Real weight = constantWeight(r, horizon);

				Real V_j = volume[j];

				const Real scale = V_i*V_j;

				const Real sw_ij = dt*dt*scale*weight;

				Matrix PK2_ij1 = 24000000 * 2.0 * Matrix::identityMatrix();
				Matrix PK2_ij2 = 24000000 * 2.0 * Matrix::identityMatrix();

				Coord rest_dir_ij = F_i *(y_rest_i - np_j.pos);

				rest_dir_ij = rest_dir_ij.norm() > EPSILON ? rest_dir_ij.normalize() : Coord(0, 0, 0);

				Coord ds_ij = sw_ij * PK2_ij1*y_pre_j + sw_ij * PK2_ij2*r*rest_dir_ij;
				Coord ds_ji = sw_ij * PK2_ij1*y_pre_i - sw_ij * PK2_ij2*r*rest_dir_ij;

				Matrix mat_ij = sw_ij * PK2_ij1;

				ds_ij *= frac_i;
				ds_ji *= frac_i;

				mat_ij *= frac_i;

				source_i += ds_ij;
				mat_i += mat_ij;

				atomicAdd(&source[j][0], ds_ji[0]);
				atomicAdd(&source[j][1], ds_ji[1]);
				atomicAdd(&source[j][2], ds_ji[2]);

				atomicAdd(&A[j](0, 0), mat_ij(0, 0));
				atomicAdd(&A[j](0, 1), mat_ij(0, 1));
				atomicAdd(&A[j](0, 2), mat_ij(0, 2));
				atomicAdd(&A[j](1, 0), mat_ij(1, 0));
				atomicAdd(&A[j](1, 1), mat_ij(1, 1));
				atomicAdd(&A[j](1, 2), mat_ij(1, 2));
				atomicAdd(&A[j](2, 0), mat_ij(2, 0));
				atomicAdd(&A[j](2, 1), mat_ij(2, 1));
				atomicAdd(&A[j](2, 2), mat_ij(2, 2));
			}
		}

		atomicAdd(&source[pId][0], source_i[0]);
		atomicAdd(&source[pId][1], source_i[1]);
		atomicAdd(&source[pId][2], source_i[2]);

		atomicAdd(&A[pId](0, 0), mat_i(0, 0));
		atomicAdd(&A[pId](0, 1), mat_i(0, 1));
		atomicAdd(&A[pId](0, 2), mat_i(0, 2));
		atomicAdd(&A[pId](1, 0), mat_i(1, 0));
		atomicAdd(&A[pId](1, 1), mat_i(1, 1));
		atomicAdd(&A[pId](1, 2), mat_i(1, 2));
		atomicAdd(&A[pId](2, 0), mat_i(2, 0));
		atomicAdd(&A[pId](2, 1), mat_i(2, 1));
		atomicAdd(&A[pId](2, 2), mat_i(2, 2));
	}


	/**
	 * @brief guarantee i and j are neighbors of each, otherwise, weird behaviors can be notices. Note there still have some bugs.
	 * 
	 * @tparam Real 
	 * @tparam Coord 
	 * @tparam Matrix 
	 * @tparam NPair 
	 * @param source 
	 * @param A 
	 * @param y_pre 
	 * @param matU 
	 * @param matV 
	 * @param eigen 
	 * @param validOfK 
	 * @param F 
	 * @param coefficients 
	 * @param restShapes 
	 * @param horizon 
	 * @param volume 
	 * @param dt 
	 * @param type 
	 * @param fraction 
	 * @return __global__ 
	 */
	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_JacobiStepSymmetric(
		GArray<Coord> source,
		GArray<Matrix> A,
		GArray<Coord> y_pre,
		GArray<Matrix> matU,
		GArray<Matrix> matV,
		GArray<Coord> eigen,
		GArray<bool> validOfK,
		GArray<Matrix> F,
		GArray<Real> coefficients,
		NeighborList<NPair> restShapes,
		Real horizon,
		GArray<Real> volume,
		Real dt,
		EnergyType type,
		GArray<Real> fraction)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_pre.size()) return;

#ifdef STVK_MODEL
		StVKModel<Real, Matrix>  model;
#endif
#ifdef LINEAR_MODEL
		LinearModel<Real, Matrix>  model;
#endif
#ifdef NEOHOOKEAN_MODEL
		NeoHookeanModel<Real, Matrix>  model;
#endif
#ifdef XU_MODEL
		XuModel<Real, Matrix>  model;
#endif
		//StVKModel<Real, Matrix> model;

		Real lambda_i1 = eigen[pId][0];
		Real lambda_i2 = eigen[pId][1];
		Real lambda_i3 = eigen[pId][2];

		Matrix U_i = matU[pId];
		Matrix V_i = matV[pId];
		Matrix PK1_i = U_i * model.getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3)*U_i.transpose();
		Matrix PK2_i = U_i * model.getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3)*V_i.transpose();

		Real Vol_i = volume[pId];
		Matrix F_i = F[pId];

		Coord y_pre_i = y_pre[pId];
		Coord y_rest_i = restShapes.getElement(pId, 0).pos;

		bool K_valid_i = validOfK[pId];

		Matrix mat_i(0);
		Coord source_i(0);

		int size_i = restShapes.getNeighborSize(pId);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord y_pre_j = y_pre[j];
			Real r = (np_j.pos - y_rest_i).norm();

			if (r > EPSILON)
			{
				Real weight_ij = constantWeight(r, horizon);
				Real Vol_j = volume[j];

				Coord y_ij = y_pre_i - y_pre_j;
				Real lambda = y_ij.norm() / r;

				const Real scale = Vol_i * Vol_j;
				const Real sw_ij = dt * dt*scale*weight_ij;

				bool K_valid_j = validOfK[j];

				Matrix U_j = matU[j];
				Matrix V_j = matV[j];

				Real lambda_j1 = eigen[j][0];
				Real lambda_j2 = eigen[j][1];
				Real lambda_j3 = eigen[j][2];

				Matrix PK1_i = K_valid_i ? PK1_i : model.getStressTensorPositive(lambda, lambda, lambda);
				Matrix PK2_i = K_valid_i ? PK2_i : model.getStressTensorNegative(lambda, lambda, lambda);

				Matrix PK1_j = U_j * model.getStressTensorPositive(lambda_j1, lambda_j2, lambda_j3)*U_j.transpose();
				Matrix PK2_j = U_j * model.getStressTensorNegative(lambda_j1, lambda_j2, lambda_j3)*V_j.transpose();

				PK1_j = K_valid_j ? PK1_j : model.getStressTensorPositive(lambda, lambda, lambda);
				PK2_j = K_valid_j ? PK2_j : model.getStressTensorNegative(lambda, lambda, lambda);

				Coord dir_ij = lambda > EPSILON ? y_ij.normalize() : Coord(1, 0, 0);

				Coord x_ij_i = K_valid_i ? y_rest_i - np_j.pos : dir_ij * (y_rest_i - np_j.pos).norm();
				Coord x_ij_j = K_valid_j ? y_rest_i - np_j.pos : dir_ij * (y_rest_i - np_j.pos).norm();

				Coord ds_i = sw_ij * PK1_i*y_pre_j + sw_ij * PK2_i*x_ij_i;
				Coord ds_j = sw_ij * PK1_j*y_pre_j + sw_ij * PK2_j*x_ij_j;

				Matrix mat_ij = sw_ij * (PK1_i + PK1_j);

				source_i += (ds_i + ds_j);
				mat_i += mat_ij;
			}
		}
		source[pId] += source_i * fraction[pId];
		A[pId] += mat_i * fraction[pId];
	}


	//-test: to find generalized inverse of all deformation gradient matrices
	// these deformation gradients are mat3x3, may be singular
	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_ComputeF(
		GArray<Matrix> F,
		GArray<Coord> eigens,
		GArray<Matrix> invK,
		GArray<bool> validOfK,
		GArray<Matrix> matU,
		GArray<Matrix> matV,
		GArray<Matrix> Rots,
		GArray<Coord> position,
		NeighborList<NPair> restShapes,
		Real horizon)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		int size_i = restShapes.getNeighborSize(pId);

		Real total_weight = Real(0);
		Matrix matL_i(0);
		Matrix matK_i(0);

#ifdef DEBUG_INFO
		if (pId == 497)
		{
			printf("Position in HM_ComputeF %d: %f %f %f \n", pId, position[pId][0], position[pId][1], position[pId][2]);
		}
#endif // DEBUG_INFO
		//printf("%d \n", size_i);

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

#ifdef DEBUG_INFO
		printf("Max distance %d: %f \n", pId, maxDist);
#endif // DEBUG_INFO

		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			// 			if (pId == 4)
			// 			{
			// 				printf("Rest %d: %f %f %f \n", ne, rest_pos_j[0], rest_pos_j[1], rest_pos_j[2]);
			// 			}

			if (r > EPSILON)
			{
				Real weight = Real(1);

				Coord p = (position[j] - position[pId]) / maxDist;
				Coord q = (rest_pos_j - rest_pos_i) / maxDist;

				matL_i(0, 0) += p[0] * q[0] * weight; matL_i(0, 1) += p[0] * q[1] * weight; matL_i(0, 2) += p[0] * q[2] * weight;
				matL_i(1, 0) += p[1] * q[0] * weight; matL_i(1, 1) += p[1] * q[1] * weight; matL_i(1, 2) += p[1] * q[2] * weight;
				matL_i(2, 0) += p[2] * q[0] * weight; matL_i(2, 1) += p[2] * q[1] * weight; matL_i(2, 2) += p[2] * q[2] * weight;

				matK_i(0, 0) += q[0] * q[0] * weight; matK_i(0, 1) += q[0] * q[1] * weight; matK_i(0, 2) += q[0] * q[2] * weight;
				matK_i(1, 0) += q[1] * q[0] * weight; matK_i(1, 1) += q[1] * q[1] * weight; matK_i(1, 2) += q[1] * q[2] * weight;
				matK_i(2, 0) += q[2] * q[0] * weight; matK_i(2, 1) += q[2] * q[1] * weight; matK_i(2, 2) += q[2] * q[2] * weight;

				total_weight += weight;

#ifdef DEBUG_INFO
				if (pId == 497)
				{
					printf("%d Neighbor %d: %f %f %f \n", pId, j, position[j][0], position[j][1], position[j][2]);
				}
#endif // DEBUG_INFO
			}
		}


		if (total_weight > EPSILON)
		{
			matL_i *= (1.0f / total_weight);
			matK_i *= (1.0f / total_weight);
		}

		Matrix R, U, D, V;
		polarDecomposition(matK_i, R, U, D, V);

#ifdef DEBUG_INFO
		if (pId == 497)
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

		Real maxK = maximum(abs(D(0, 0)), maximum(abs(D(1, 1)), abs(D(2, 2))));
		Real minK = minimum(abs(D(0, 0)), minimum(abs(D(1, 1)), abs(D(2, 2))));

		//TODO: how to deal with bad K
		bool valid_K = minK < EPSILON || maxK / minK > Real(3) ? false : true;

		validOfK[pId] = valid_K;
		//printf("Det %d: %f \n", pId, matK_i.determinant());

		Matrix F_i;
		if (valid_K)
		{
			invK[pId] = matK_i.inverse();
			F_i = matL_i * matK_i.inverse();
		}
		else
		{
			Real threshold = 0.0001f*horizon;
			D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
			D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
			D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;
			invK[pId] = V * D*U.transpose();
			F_i = matL_i * V * D*U.transpose();
		}

		//eigens[pId] = Coord(clamp(D(0, 0), Real(slimit), Real(1/ slimit)), clamp(D(1, 1), Real(slimit), Real(1 / slimit)), clamp(D(2, 2), Real(slimit), Real(1 / slimit)));

//		printf("Det %d: %f \n", pId, F_i.determinant());

		polarDecomposition(F_i, R, U, D, V);

		if (F_i.determinant() < maxDist*EPSILON)
		{
			R = Rots[pId];
		}
		else
			Rots[pId] = R;

		Coord n0 = Coord(1, 0, 0);
		Coord n1 = Coord(0, 1, 0);
		Coord n2 = Coord(0, 0, 1);

		Coord n0_rot = R * n0;
		Coord n1_rot = R * n1;
		Coord n2_rot = R * n2;

		Real l0 = n0_rot.dot(F_i*n0);
		Real l1 = n1_rot.dot(F_i*n1);
		Real l2 = n2_rot.dot(F_i*n2);

		matU[pId] = R;
		matV[pId] = Matrix::identityMatrix();

		const Real slimit = Real(0.9);

		l0 = clamp(l0, slimit, 1 / slimit);
		l1 = clamp(l1, slimit, 1 / slimit);
		l2 = clamp(l2, slimit, 1 / slimit);

		D(0, 0) = l0;
		D(1, 1) = l1;
		D(2, 2) = l2;

//		printf("Stretches %d: %f %f %f \n", pId, l0, l1, l2);

		eigens[pId] = Coord(l0, l1, l2);

		F[pId] = R * D;

#ifdef DEBUG_INFO
		//if (pId == 0)
		{
			Matrix mat_out = R.transpose()*F_i;
			printf("Rot F %d: \n %f %f %f \n %f %f %f \n %f %f %f \n\n", pId,
				mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
				mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
				mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

			printf("eigen value %d: %f %f %f \n", pId, eigens[pId][0], eigens[pId][1], eigens[pId][2]);
		}
#endif // DEBUG_INFO

		/*		//Check whether the shape is inverted, if yes, use previously computed U and V instead
				if (F_i.determinant() <= EPSILON)
				{
					U = matU[pId];
					V = matV[pId];

					Coord unit_x = Coord(1, 0, 0);
					Coord unit_y = Coord(0, 1, 0);
					Coord unit_z = Coord(0, 0, 1);

					Matrix UtFV_i = U.transpose()*F_i*V;
					Real l1 = unit_x.dot(UtFV_i*unit_x);
					Real l2 = unit_y.dot(UtFV_i*unit_y);
					Real l3 = unit_z.dot(UtFV_i*unit_z);

					l1 = l1 > 0 ? l1 : -l1;
					l2 = l2 > 0 ? l2 : -l2;
					l3 = l3 > 0 ? l3 : -l3;

					l1 = l1 < slimit ? slimit : l1;
					l2 = l2 < slimit ? slimit : l2;
					l3 = l3 < slimit ? slimit : l3;

					eigens[pId] = Coord(l1, l2, l3);

					D(0, 0) = l1;
					D(1, 1) = l2;
					D(2, 2) = l3;

		// 			U = Matrix::identityMatrix();
		// 			V = Matrix::identityMatrix();
		//
		// 			D(0, 0) = 1;
		// 			D(1, 1) = 1;
		// 			D(2, 2) = 1;

					F[pId] = U * D*V.transpose();

		#ifdef DEBUG_INFO
					//if (pId == 0)
					{
						Matrix mat_out = F[pId];
						printf("Mat singular %d: \n %f %f %f \n %f %f %f \n %f %f %f \n\n", pId,
							mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
							mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
							mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

						printf("eigen value %d: %f %f %f \n", pId, eigens[pId][0], eigens[pId][1], eigens[pId][2]);
					}
		#endif // DEBUG_INFO
				}
				else
				{
					polarDecomposition(F_i, R, U, D, V);

					D(0, 0) = clamp(D(0, 0), Real(slimit), Real(1 / slimit));
					D(1, 1) = clamp(D(1, 1), Real(slimit), Real(1 / slimit));
					D(2, 2) = clamp(D(2, 2), Real(slimit), Real(1 / slimit));

					eigens[pId] = Coord(D(0, 0), D(1, 1), D(2, 2));
					matU[pId] = U;
					matV[pId] = V;
					F[pId] = U * D*V.transpose();

		#ifdef DEBUG_INFO
					if (pId == 0)
					{
						Matrix mat_out = matU[pId];
						printf("matU %d: \n %f %f %f \n %f %f %f \n %f %f %f \n\n", pId,
							mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
							mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
							mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

						mat_out = F[pId];
						printf("F %d: \n %f %f %f \n %f %f %f \n %f %f %f \n\n", pId,
							mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
							mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
							mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

						mat_out = matV[pId];
						printf("matV %d: \n %f %f %f \n %f %f %f \n %f %f %f \n\n", pId,
							mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
							mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
							mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

						mat_out = R;
						printf("Rot %d: \n %f %f %f \n %f %f %f \n %f %f %f \n\n", pId,
							mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
							mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
							mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

						mat_out = U*V.transpose();
						printf("UVT %d: \n %f %f %f \n %f %f %f \n %f %f %f \n\n", pId,
							mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
							mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
							mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

						printf("eigen value %d: %f %f %f \n", pId, eigens[pId][0], eigens[pId][1], eigens[pId][2]);
					}
		#endif // DEBUG_INFO
				}*/
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_JacobiStepNonsymmetric(
		GArray<Coord> source,
		GArray<Matrix> A,
		GArray<Coord> y_pre,
		GArray<Matrix> matU,
		GArray<Matrix> matV,
		GArray<Coord> eigen,
		GArray<bool> validOfK,
		GArray<Matrix> F,
		NeighborList<NPair> restShapes,
		Real horizon,
		GArray<Real> volume,
		Real dt,
		EnergyType type,
		GArray<Real> fraction)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_pre.size()) return;

#ifdef STVK_MODEL
		StVKModel<Real, Matrix>  model;
#endif
#ifdef LINEAR_MODEL
		LinearModel<Real, Matrix>  model;
#endif
#ifdef NEOHOOKEAN_MODEL
		NeoHookeanModel<Real, Matrix>  model;
#endif
#ifdef XU_MODEL
		XuModel<Real, Matrix>  model;
#endif

		Real frac_i = fraction[pId];


		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		int size_i = restShapes.getNeighborSize(pId);

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

		Real kappa = 16 / (15 * M_PI*maxDist*maxDist*maxDist*maxDist*maxDist);

		Real lambda_i1 = eigen[pId][0];
		Real lambda_i2 = eigen[pId][1];
		Real lambda_i3 = eigen[pId][2];

		Matrix U_i = matU[pId];
		Matrix V_i = matV[pId];
		Matrix PK1_i = U_i * model.getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3)*U_i.transpose();
		Matrix PK2_i = U_i * model.getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3)*V_i.transpose();

		Real Vol_i = volume[pId];
		Matrix F_i = F[pId];

		Coord y_pre_i = y_pre[pId];
		Coord y_rest_i = restShapes.getElement(pId, 0).pos;

		bool K_valid = validOfK[pId];

		Matrix mat_i(0);
		Coord source_i(0);

// 		if (validOfK[pId])
// 		{
// 			printf("Valid: %d \n", pId);
// 		}
// 		else
// 		{
// 			printf("Invalid: %d \n", pId);
// 		}

		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord y_pre_j = y_pre[j];
			Real r = (np_j.pos - y_rest_i).norm();

			if (r > EPSILON)
			{
				Real weight_ij = constantWeight(r, maxDist);
				Real Vol_j = volume[j];

				Coord y_ij = y_pre_i - y_pre_j;
				Real lambda = y_ij.norm() / r;

				const Real scale = Vol_i * Vol_j*kappa;
				const Real sw_ij = dt * dt*scale*weight_ij;

				bool K_valid_ij = K_valid & validOfK[j];

				Matrix PK1_ij = K_valid_ij ? PK1_i : model.getStressTensorPositive(lambda, lambda, lambda);
				Matrix PK2_ij = K_valid_ij ? PK2_i : model.getStressTensorNegative(lambda, lambda, lambda);

				Coord dir_ij = lambda > EPSILON ? y_ij.normalize() : Coord(1, 0, 0);

				Coord x_ij = K_valid_ij ? y_rest_i - np_j.pos : dir_ij * (y_rest_i - np_j.pos).norm();

				Coord ds_ij = sw_ij * PK1_ij*y_pre_j + sw_ij * PK2_ij*x_ij;
				Coord ds_ji = sw_ij * PK1_ij*y_pre_i - sw_ij * PK2_ij*x_ij;

				Matrix mat_ij = sw_ij * PK1_ij;

				ds_ij *= frac_i;
				ds_ji *= frac_i;

				mat_ij *= frac_i;

				source_i += ds_ij;
				mat_i += mat_ij;

				atomicAdd(&source[j][0], ds_ji[0]);
				atomicAdd(&source[j][1], ds_ji[1]);
				atomicAdd(&source[j][2], ds_ji[2]);

				atomicAdd(&A[j](0, 0), mat_ij(0, 0));
				atomicAdd(&A[j](0, 1), mat_ij(0, 1));
				atomicAdd(&A[j](0, 2), mat_ij(0, 2));
				atomicAdd(&A[j](1, 0), mat_ij(1, 0));
				atomicAdd(&A[j](1, 1), mat_ij(1, 1));
				atomicAdd(&A[j](1, 2), mat_ij(1, 2));
				atomicAdd(&A[j](2, 0), mat_ij(2, 0));
				atomicAdd(&A[j](2, 1), mat_ij(2, 1));
				atomicAdd(&A[j](2, 2), mat_ij(2, 2));
			}
		}

		atomicAdd(&source[pId][0], source_i[0]);
		atomicAdd(&source[pId][1], source_i[1]);
		atomicAdd(&source[pId][2], source_i[2]);

		atomicAdd(&A[pId](0, 0), mat_i(0, 0));
		atomicAdd(&A[pId](0, 1), mat_i(0, 1));
		atomicAdd(&A[pId](0, 2), mat_i(0, 2));
		atomicAdd(&A[pId](1, 0), mat_i(1, 0));
		atomicAdd(&A[pId](1, 1), mat_i(1, 1));
		atomicAdd(&A[pId](1, 2), mat_i(1, 2));
		atomicAdd(&A[pId](2, 0), mat_i(2, 0));
		atomicAdd(&A[pId](2, 1), mat_i(2, 1));
		atomicAdd(&A[pId](2, 2), mat_i(2, 2));

// 		source[pId] += source_i * fraction[pId];
// 		A[pId] += mat_i * fraction[pId];
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_JacobiStepDegenerate(
		GArray<Coord> source,
		GArray<Matrix> A,
		GArray<Coord> y_next,
		GArray<Coord> y_pre,
		GArray<Coord> y_old,
		GArray<Matrix> matU,
		GArray<Matrix> matV,
		GArray<Coord> eigen,
		GArray<Matrix> F,
		GArray<Real> coefficients,
		NeighborList<NPair> restShapes,
		Real horizon,
		GArray<Real> volume,
		Real dt,
		EnergyType type,
		GArray<Real> fraction)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

#ifdef STVK_MODEL
		StVKModel<Real, Matrix>  model;
#endif
#ifdef LINEAR_MODEL
		LinearModel<Real, Matrix>  model;
#endif
#ifdef NEOHOOKEAN_MODEL
		NeoHookeanModel<Real, Matrix>  model;
#endif
#ifdef XU_MODEL
		XuModel<Real, Matrix>  model;
#endif

		Real Vol_i = volume[pId];
		Matrix F_i = F[pId];

		Coord y_pre_i = y_pre[pId];
		Coord y_rest_i = restShapes.getElement(pId, 0).pos;

		Real ceof_i = coefficients[pId];

		Matrix mat_i(0);
		Coord source_i(0);

		int size_i = restShapes.getNeighborSize(pId);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord y_pre_j = y_pre[j];
			Real r = (np_j.pos - y_rest_i).norm();

			Real ceof_j = coefficients[j];

			if (r > EPSILON)
			{
				Real weight = constantWeight(r, horizon);

				Real Vol_j = volume[j];
				Matrix F_j = F[j];

				const Real scale = 0.5f*Vol_i * Vol_j*(ceof_i + ceof_j);

				const Real sw_ij = dt * dt*scale*weight;

				Coord y_ij = y_pre_i - y_pre_j;
				Real lambda = y_ij.norm() / r;

				Coord dir_ij = y_ij;
				if (lambda > EPSILON)
				{
					dir_ij.normalize();
				}
				else
				{
					dir_ij = Coord(1, 0, 0);
				}


				Matrix U_j = Matrix::identityMatrix();
				Matrix V_j = Matrix::identityMatrix();
				Matrix PK1_j = U_j * model.getStressTensorPositive(lambda, lambda, lambda)*U_j.transpose();
				Matrix PK2_j = U_j * model.getStressTensorNegative(lambda, lambda, lambda)*V_j.transpose();
				Matrix PK1_ij = PK1_j + PK1_j;
				Matrix PK2_ij = PK2_j + PK2_j;


// 				if (pId == 0)
// 				{
// 					Matrix mat_out = PK1_ij;
// 					printf("Mat: \n %f %f %f \n %f %f %f \n %f %f %f \n\n",
// 						mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
// 						mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
// 						mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));
// 				}

				Coord x_ij = y_rest_i - np_j.pos;

				source_i += sw_ij * PK2_ij*dir_ij*x_ij.norm();
				source_i += sw_ij * PK1_ij*y_pre_j;

				mat_i += sw_ij * PK1_ij;
			}
		}

		source[pId] += source_i * fraction[pId];
		A[pId] += mat_i * fraction[pId];
	}


	template <typename Real, typename Coord, typename Matrix>
	__global__ void HM_ComputeNextPosition(
		GArray<Coord> y_next,
		GArray<Coord> y_pre,
		GArray<Coord> y_old,
		GArray<Real> volume,
		GArray<Coord> source,
		GArray<Attribute> atts,
		GArray<Matrix> A,
		EnergyType type)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

#ifdef STVK_MODEL
		StVKModel<Real, Matrix>  model;
#endif
#ifdef LINEAR_MODEL
		LinearModel<Real, Matrix>  model;
#endif
#ifdef NEOHOOKEAN_MODEL
		NeoHookeanModel<Real, Matrix>  model;
#endif
#ifdef XU_MODEL
		XuModel<Real, Matrix>  model;
#endif

		Real mass_i = volume[pId] * model.density;

		Matrix mat_i = A[pId] + mass_i*Matrix::identityMatrix();
		Coord src_i = source[pId] + mass_i*y_old[pId];
			
		Attribute att = atts[pId];
		if (att.IsDynamic())
		{
			y_next[pId] = mat_i.inverse()*src_i;
		}
		else
		{
			y_next[pId] = y_pre[pId];
		}
		
// 		if (pId == 487)
// 		{
// 			printf("Y 487: %f %f %f \n", y_next[pId][0], y_next[pId][1], y_next[pId][2]);
// 		}

// 			if (pId == 0)
// 			{
// 				Matrix mat_out = A[pId];
// 				printf("Mat: \n %f %f %f \n %f %f %f \n %f %f %f \n\n",
// 					mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
// 					mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
// 					mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));
// 
// 				printf("mass_i: \n %f \n\n", mass_i);
// 
// 				printf("source: %f %f %f \n", source[pId][0], source[pId][1], source[pId][2]);
// 
// 				printf("y_old: %f %f %f \n", y_old[pId][0], y_old[pId][1], y_old[pId][2]);
// 
// 				printf("y_next: %f %f %f \n", y_next[pId][0], y_next[pId][1], y_next[pId][2]);
// 			}
	}

	template <typename Real, typename Coord, typename Matrix>
	__global__ void HM_ComputeResidual(
		GArray<Coord> residual,
		GArray<Coord> y_current,
		GArray<Coord> y_old,
		GArray<Real> volume,
		GArray<Coord> source,
		GArray<Matrix> A,
		EnergyType type)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_current.size()) return;

#ifdef STVK_MODEL
		StVKModel<Real, Matrix>  model;
#endif
#ifdef LINEAR_MODEL
		LinearModel<Real, Matrix>  model;
#endif
#ifdef NEOHOOKEAN_MODEL
		NeoHookeanModel<Real, Matrix>  model;
#endif
#ifdef XU_MODEL
		XuModel<Real, Matrix>  model;
#endif

		Real mass_i = volume[pId] * model.density;

		Matrix mat_i = A[pId] + mass_i * Matrix::identityMatrix();
		Coord src_i = source[pId] + mass_i * y_old[pId];

		residual[pId] = mat_i * y_current[pId] - src_i;
	}


	template <typename Real, typename Coord, typename Matrix>
	__global__ void HM_ComputeGradC(
		GArray<Coord> gradC,
		GArray<Coord> y_delta,
		GArray<Real> volume,
		GArray<Coord> source,
		GArray<Matrix> A,
		EnergyType type)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_delta.size()) return;

#ifdef STVK_MODEL
		StVKModel<Real, Matrix>  model;
#endif
#ifdef LINEAR_MODEL
		LinearModel<Real, Matrix>  model;
#endif
#ifdef NEOHOOKEAN_MODEL
		NeoHookeanModel<Real, Matrix>  model;
#endif
#ifdef XU_MODEL
		XuModel<Real, Matrix>  model;
#endif

		Real mass_i = volume[pId] * model.density;

		Matrix mat_i = A[pId] + mass_i * Matrix::identityMatrix();
		Coord src_i = source[pId];

		gradC[pId] = mat_i * y_delta[pId] - src_i;
	}


	template <typename Matrix>
	__global__ void HM_InitRotation(
		GArray<Matrix> U,
		GArray<Matrix> V)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= U.size()) return;

		U[pId] = Matrix::identityMatrix();
		V[pId] = Matrix::identityMatrix();
	}

	template<typename TDataType>
	bool HyperelasticityModule_test<TDataType>::initializeImpl()
	{
		m_F.resize(this->inPosition()->getElementCount());
		m_invF.resize(this->inPosition()->getElementCount());
		m_invK.resize(this->inPosition()->getElementCount());
		m_validOfK.resize(this->inPosition()->getElementCount());

		m_eigenValues.resize(this->inPosition()->getElementCount());
		m_matU.resize(this->inPosition()->getElementCount());
		m_matV.resize(this->inPosition()->getElementCount());

		m_energy.resize(this->inPosition()->getElementCount());
		m_alpha.resize(this->inPosition()->getElementCount());
		m_gradient.resize(this->inPosition()->getElementCount());

		y_next.resize(this->inPosition()->getElementCount());
		y_current.resize(this->inPosition()->getElementCount());
		y_residual.resize(this->inPosition()->getElementCount());
		y_gradC.resize(this->inPosition()->getElementCount());

		m_source.resize(this->inPosition()->getElementCount());
		m_A.resize(this->inPosition()->getElementCount());

		m_position_old.resize(this->inPosition()->getElementCount());

		m_fraction.resize(this->inPosition()->getElementCount());

		m_bFixed.resize(this->inPosition()->getElementCount());
		m_fixedPos.resize(this->inPosition()->getElementCount());
		this->m_points_move_type.resize(this->inPosition()->getElementCount());

//		initializeVolume();

		cuExecute(m_matU.size(),
			HM_InitRotation,
			m_matU,
			m_matV);

		m_reduce = Reduction<Real>::Create(this->inPosition()->getElementCount());
		m_alg = Arithmetic<Real>::Create(this->inPosition()->getElementCount());

		return ElasticityModule::initializeImpl();
	}

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::begin()
	{
		int num_new = this->inPosition()->getElementCount();
		if (num_new == m_F.size())
			return;

		m_F.resize(this->inPosition()->getElementCount());
		m_invF.resize(this->inPosition()->getElementCount());
		m_invK.resize(this->inPosition()->getElementCount());
		m_validOfK.resize(this->inPosition()->getElementCount());

		m_eigenValues.resize(this->inPosition()->getElementCount());
		m_matU.resize(this->inPosition()->getElementCount());
		m_matV.resize(this->inPosition()->getElementCount());

		m_energy.resize(this->inPosition()->getElementCount());
		m_alpha.resize(this->inPosition()->getElementCount());
		m_gradient.resize(this->inPosition()->getElementCount());

		y_next.resize(this->inPosition()->getElementCount());
		y_current.resize(this->inPosition()->getElementCount());
		y_residual.resize(this->inPosition()->getElementCount());
		y_gradC.resize(this->inPosition()->getElementCount());

		m_source.resize(this->inPosition()->getElementCount());
		m_A.resize(this->inPosition()->getElementCount());

		m_position_old.resize(this->inPosition()->getElementCount());

		m_fraction.resize(this->inPosition()->getElementCount());

		m_bFixed.resize(this->inPosition()->getElementCount());
		m_fixedPos.resize(this->inPosition()->getElementCount());
		this->m_points_move_type.resize(this->inPosition()->getElementCount());

//		initializeVolume();

		cuExecute(m_matU.size(),
			HM_InitRotation,
			m_matU,
			m_matV);

		ElasticityModule::begin();
	}


	template <typename Real>
	__global__ void HM_InitVolume(
		GArray<Real> volume
	) 
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= volume.size()) return;

		volume[pId] = Real(1);
	}
	

// 	template<typename TDataType>
// 	void HyperelasticityModule_test<TDataType>::initializeVolume()
// 	{
// 		int numOfParticles = this->inPosition()->getElementCount();
// 		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);
// 
// 		HM_InitVolume << <pDims, BLOCK_SIZE >> > (m_volume);
// 	}

	template <typename Coord, typename Matrix>
	__global__ void HM_AdjustFixedPos_rotate(
		int adjust_type,
		GArray<int> m_points_move_type,
		GArray<Coord> fixedPos,
		Matrix rotate_mat
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= fixedPos.size()) return;

		if (m_points_move_type[pId] == adjust_type)
		{
			fixedPos[pId] = rotate_mat * fixedPos[pId];
			//fixedPos[pId] += Coord(0.0075, 0, 0);
		}
	}


	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticity()
	{
		solveElasticityImplicit();
		//ElasticityModule::solveElasticity();
	}

	int ind_num = 0;

	template <typename Coord, typename Matrix>
	__global__ void HM_RotateInitPos(
		GArray<Coord> position)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		RandNumber gen(pId);

		Matrix rotM(0);
		float theta = 3.1415926 / 4.0f;
		rotM(0, 0) = cos(theta);
		rotM(0, 1) = -sin(theta);
		rotM(1, 0) = sin(theta);
		rotM(1, 1) = cos(theta);
// 		rotM(0, 0) = sin(theta);
// 		rotM(0, 1) = cos(theta);
// 		rotM(1, 0) = cos(theta);
// 		rotM(1, 1) = -sin(theta);
		rotM(2, 2) = 1.0f;
		Coord origin = position[0];
		//position[pId] = origin + rotM*(position[pId] - origin);
		position[pId][0] += 0.01*(gen.Generate() - 0.5);
		position[pId][1] += 0.01*(gen.Generate() - 0.5);
		position[pId][2] += 0.01*(gen.Generate() - 0.5);
//		position[pId][1] = - position[pId][1] + 0.1;
//		position[pId][1] += (0.5 - position[pId][1]) + 0.5;

// 		if (pId >= FIXEDNUM && pId < position.size() - FIXEDNUM)
// 		{
// 			position[pId] -= Coord(0.0035, 0, 0);
// 		}
	}

	template <typename Coord>
	__global__ void HM_ComputeGradient(
		GArray<Coord> grad,
		GArray<Coord> y_pre,
		GArray<Coord> y_next)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		grad[pId] = y_next[pId] - y_pre[pId];

// 		if (pId == grad.size() - FIXEDNUM - 2)
// // 		if (grad[pId].norm() > 0.00001)
//  			printf("Thread ID %d: %f, %f, %f \n", pId, grad[pId][0], grad[pId][1], grad[pId][2]);
	}

	template <typename Real, typename Coord>
	__global__ void HM_ComputeCurrentPosition(
		GArray<Coord> grad,
		GArray<Coord> y_current,
		GArray<Coord> y_next,
		GArray<Real> alpha)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		y_next[pId] = y_current[pId] + alpha[pId]*grad[pId];
	}

	template <typename Real, typename Coord>
	__global__ void HM_ComputeCurrentPosition(
		GArray<Coord> grad,
		GArray<Coord> y_current,
		GArray<Coord> y_next,
		Real alpha)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		y_next[pId] = y_current[pId] + alpha * grad[pId];
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_Compute1DEnergy(
		GArray<Real> energy,
		GArray<Coord> pos_current,
		GArray<Matrix> F,
		GArray<Real> volume,
		GArray<bool> validOfK,
		GArray<Coord> eigenValues,
		NeighborList<NPair> restShapes,
		EnergyType type)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= energy.size()) return;

		Coord pos_current_i = pos_current[pId];

		Real totalEnergy = 0.0f;// (pos_current_i - pos_pre_i).normSquared();
		Real V_i = volume[pId];

#ifdef STVK_MODEL
		StVKModel<Real, Matrix>  model;
#endif
#ifdef LINEAR_MODEL
		LinearModel<Real, Matrix>  model;
#endif
#ifdef NEOHOOKEAN_MODEL
		NeoHookeanModel<Real, Matrix>  model;
#endif
#ifdef XU_MODEL
		XuModel<Real, Matrix>  model;
#endif

		int size_i = restShapes.getNeighborSize(pId);

		Coord rest_pos_i = restShapes.getElement(pId, 0).pos;
		Coord eigen_value_i = eigenValues[pId];
		bool valid_i = validOfK[pId];

		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
			Coord pos_current_j = pos_current[j];
			Real r = (np_j.pos - rest_pos_i).norm();

			Real V_j = volume[j];

			if (r > EPSILON)
			{
				if (valid_i)
				{
					totalEnergy += V_j * model.getEnergy(eigen_value_i[0], eigen_value_i[1], eigen_value_i[2]);
				}
				else
				{
					Real stretch = (pos_current_j - pos_current_i).normSquared() / (r*r);
					totalEnergy += V_j * model.getEnergy(stretch, stretch, stretch);
				}
			}
		}

		energy[pId] = totalEnergy * V_i;
	}

	template <typename Coord>
	__global__ void HM_Chebyshev_Acceleration(GArray<Coord> next_X, GArray<Coord> X, GArray<Coord> prev_X, float omega)
	{
		int pId = blockDim.x * blockIdx.x + threadIdx.x;
		if (pId >= prev_X.size())	return;

		next_X[pId] = (next_X[pId] - X[pId])*0.666 + X[pId];

		next_X[pId] = omega*(next_X[pId] - prev_X[pId]) + prev_X[pId];
	}

	template <typename Real, typename Coord>
	__global__ void HM_DOT(
		GArray<Real> gradSquare,
		GArray<Real> gradResidual,
		GArray<Coord> gradC,
		GArray<Coord> residual)
	{
		int pId = blockDim.x * blockIdx.x + threadIdx.x;
		if (pId >= residual.size())	return;

		Coord grd = gradC[pId];
		Coord res = residual[pId];

		gradSquare[pId] = grd.dot(grd);

		gradResidual[pId] = res.dot(grd);
	}

	template <typename Coord, typename NPair>
	__global__ void HM_AccumulateGradient(
		GArray<Coord> accumulate,
		GArray<Coord> gradient,
		NeighborList<NPair> restShapes)
	{
		int pId = blockDim.x * blockIdx.x + threadIdx.x;
		if (pId >= accumulate.size())	return;

		Coord acc_i = gradient[pId];
		int size_i = restShapes.getNeighborSize(pId);
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;

			acc_i -= gradient[j];
		}

		accumulate[pId] = acc_i;
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void HM_ComputeStepLength(
		GArray<Real> stepLength,
		GArray<Coord> gradient,
		GArray<Real> volume,
		GArray<Matrix> A,
		GArray<Real> energy,
		NeighborList<NPair> restShapes)
	{
		int pId = blockDim.x * blockIdx.x + threadIdx.x;
		if (pId >= stepLength.size())	return;

#ifdef STVK_MODEL
		StVKModel<Real, Matrix>  model;
#endif
#ifdef LINEAR_MODEL
		LinearModel<Real, Matrix>  model;
#endif
#ifdef NEOHOOKEAN_MODEL
		NeoHookeanModel<Real, Matrix>  model;
#endif
#ifdef XU_MODEL
		XuModel<Real, Matrix>  model;
#endif

		Real mass_i = volume[pId] * model.density;
		Real energy_i = energy[pId];
		Coord grad_i = (mass_i*Matrix::identityMatrix() + A[pId])*gradient[pId];

		Real alpha = grad_i.norm() < EPSILON ? Real(0) : energy_i / grad_i.normSquared();

	//	printf("%d step length: %f; Energy: %f \n", pId, alpha, energy_i);

		alpha = alpha < 0 ? Real(0) : alpha;
		alpha = alpha > 1 ? Real(1) : alpha;

		alpha /= Real(1 + restShapes.getNeighborSize(pId));

		//printf("%d step length: %f \n", pId, alpha);

		stepLength[pId] = alpha;
	}

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticityImplicit()
	{
// 		auto d_ele = this->inRestShape()->getValue().getElements();
// 		auto d_index = this->inRestShape()->getValue().getIndex();
// 
// 		CArray<NPair> h_ele;
// 		CArray<int> h_index;
// 
// 		h_ele.resize(d_ele.size());
// 		h_index.resize(d_index.size());
// 
// 		Function1Pt::copy(h_ele, d_ele);
// 		Function1Pt::copy(h_index, d_index);
// 
// 		int t = 0;
// 		for (int i = 0; i < h_ele.size(); i++)
// 		{
// 			auto ele = h_ele[i];
// 			t++;
// 		}


		int numOfParticles = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		this->m_weights.reset();

		Log::sendMessage(Log::User, "\n \n \n \n *************solver start!!!***************");

		/**************************** Jacobi method ************************************************/
		// initialize y_now, y_next_iter
		Function1Pt::copy(y_current, this->inPosition()->getValue());
		Function1Pt::copy(y_next, this->inPosition()->getValue());
		Function1Pt::copy(m_position_old, this->inPosition()->getValue());

		// do Jacobi method Loop
		bool convergeFlag = false; // converge or not
		int iterCount = 0;

		Real omega;
		Real alpha = 1.0f;
		while (iterCount < this->getIterationNumber()) {

			m_source.reset();
			m_A.reset();

			HM_ComputeF << <pDims, BLOCK_SIZE >> > (
				m_F,
				m_eigenValues,
				m_invK,
				m_validOfK,
				m_matU,
				m_matV,
				this->inRotation()->getValue(),
				y_current,
				this->inRestShape()->getValue(),
				this->inHorizon()->getValue());
			cuSynchronize();

			HM_Blend << <pDims, BLOCK_SIZE >> > (
				m_fraction,
				m_eigenValues);

			HM_JacobiLinear << <pDims, BLOCK_SIZE >> > (
				m_source,
				m_A,
				y_current,
				m_F,
				this->inRestShape()->getValue(),
				this->inVolume()->getValue(),
				this->inHorizon()->getValue(),
				this->getParent()->getDt(),
				m_fraction);

			HM_JacobiStepNonsymmetric << <pDims, BLOCK_SIZE >> > (
				m_source,
				m_A,
				y_current,
				m_matU,
				m_matV,
				m_eigenValues,
				m_validOfK,
				m_F,
				this->inRestShape()->getValue(),
				this->inHorizon()->getValue(),
				this->inVolume()->getValue(), 
				this->getParent()->getDt(),
				m_energyType,
				m_fraction);

			cuExecute(y_next.size(),
				HM_ComputeNextPosition,
				y_next,
				y_current,
				m_position_old,
				this->inVolume()->getValue(),
				m_source,
				this->inAttribute()->getValue(),
				m_A,
				m_energyType);

			//sub steping
			cuExecute(m_gradient.size(),
				HM_ComputeGradient,
				m_gradient,
				y_current,
				y_next);


			cuExecute(m_energy.size(),
				HM_Compute1DEnergy,
				m_energy,
				y_current,
				m_F,
				this->inVolume()->getValue(),
				m_validOfK,
				m_eigenValues,
				this->inRestShape()->getValue(),
				m_energyType);

			if (this->isAlphaCompute) {
				cuExecute(m_alpha.size(),
					HM_ComputeStepLength,
					m_alpha,
					m_gradient,
					this->inVolume()->getValue(),
					m_A,
					m_energy,
					this->inRestShape()->getValue());

				cuExecute(m_gradient.size(),
					HM_ComputeCurrentPosition,
					m_gradient,
					y_current,
					y_next,
					m_alpha);
			}
			else {
				alpha = Real(1.0);
				cuExecute(m_gradient.size(),
					HM_ComputeCurrentPosition,
					m_gradient,
					y_current,
					y_next,
					alpha);
			}
			
			if (this->isChebyshevAcce) {
				omega = Real(1.0);
				/*cuExecute(y_next.size(),
					HM_Chebyshev_Acceleration,
					y_next,
					y_current,
					y_pre,
					omega);*/
			}
			else {
				;
			}

// 			GArray<Coord> gradient_cpy;
// 			gradient_cpy.resize(m_gradient.size());
// 			Function1Pt::copy(gradient_cpy, m_gradient);
// 			cuExecute(m_gradient.size(),
// 				HM_AccumulateGradient,
// 				m_gradient,
// 				gradient_cpy,
// 				this->inRestShape()->getValue());
// 			gradient_cpy.release();

/*			alpha *= Real(2);
			alpha = alpha > 1.0 ? 1.0 : alpha;
			alpha = alpha < 0.0 ? 0.0 : alpha;

			HM_ComputeCurrentPosition << <pDims, BLOCK_SIZE >> > (
				m_gradient,
				y_current,
				y_next,
				alpha);

			//stepsize adjustment
			Real totalE_current;
			Real totalE_next;
			getEnergy(totalE_current, y_current);
			getEnergy(totalE_next, y_next);
			
			//if (totalE_next > totalE_current)
			{
				cuExecute(y_next.size(),
					HM_ComputeResidual,
					y_residual,
					y_current,
					m_position_old,
					m_volume,
					m_source,
					m_A,
					m_energyType);

				GArray<Matrix> tmpA;
				GArray<Coord> tmpSource;
				tmpA.resize(y_current.size());	tmpA.reset();
				tmpSource.resize(y_current.size());	tmpSource.reset();
				HM_JacobiStepSymmetric << <pDims, BLOCK_SIZE >> > (
					tmpSource,
					tmpA,
					m_gradient,
					m_matU,
					m_matV,
					m_eigenValues,
					m_validOfK,
					m_F,
					this->m_bulkCoefs,
					this->inRestShape()->getValue(),
					this->inHorizon()->getValue(),
					m_volume, this->getParent()->getDt(),
					m_energyType,
					m_fraction);

				cuExecute(y_next.size(),
					HM_ComputeGradC,
					y_gradC,
					m_gradient,
					m_volume,
					tmpSource,
					tmpA,
					m_energyType);

				tmpSource.release();
				tmpA.release();

				GArray<Real> gradSquare;
				GArray<Real> gradResidual;
				gradSquare.resize(y_current.size());
				gradResidual.resize(y_current.size());

				cuExecute(y_current.size(),
					HM_DOT,
					gradSquare,
					gradResidual,
					y_gradC,
					y_residual);

				Real denom = m_reduce->accumulate(gradSquare.begin(), gradSquare.size());
				Real numer = m_reduce->accumulate(gradResidual.begin(), gradResidual.size());

				gradSquare.release();
				gradResidual.release();

				Real alpha = denom < EPSILON ? 1.0 : numer / denom;

				printf("Alpha: %f \n", alpha);

				alpha = alpha > 1.0 ? 1.0 : alpha;
				alpha = alpha < 0.0 ? 0.0 : alpha;

				alpha = Real(1);

				printf("Numer: %f \n", numer);
				printf("Denom: %f \n", denom);

				HM_ComputeCurrentPosition << <pDims, BLOCK_SIZE >> > (
					m_gradient,
					y_current,
					y_next,
					alpha);

				//printf("Current: %f Next: %f \n", totalE_current, totalE_next);

				getEnergy(totalE_next, y_next);
				//stepping
				int step = 0;
				while (totalE_next > totalE_current && step < 50)
					//if(true)
				{
					step++;
					alpha *= 0.5;

					HM_ComputeCurrentPosition << <pDims, BLOCK_SIZE >> > (
						m_gradient,
						y_current,
						y_next,
						alpha);

					getEnergy(totalE_next, y_next);
				}

// 				alpha *= 0.5;
// 				HM_ComputeCurrentPosition << <pDims, BLOCK_SIZE >> > (
// 					m_gradient,
// 					y_current,
// 					y_next,
// 					alpha);

				printf("Substep length: %d \n", step);

				
// 				int step = 0;
// 				if (bChebyshevAccOn)
// 				{
// 					if (step <= 10)		omega = 1;
// 					else if (step == 11)	omega = 2 / (2 - rho*rho);
// 					else	omega = 4 / (4 - rho*rho*omega);
// 
// 					HM_Chebyshev_Acceleration << <pDims, BLOCK_SIZE >> > (
// 						y_next,
// 						y_current,
// 						y_pre,
// 						omega);
// 				}
			}*/
			
			if (this->var_isConvergeComputeField.getValue()) {
				CArray<Coord> y_current_host, y_next_host;
				y_current_host.resize(y_current.size());
				y_next_host.resize(y_next.size());

				Function1Pt::copy(y_current_host, y_current);
				Function1Pt::copy(y_next_host, y_next);

				int particle_num = y_current.size();
				Real delta_sum = Real(0.0);
				for (int i = 0; i < particle_num; ++i) {
					delta_sum += (y_next_host[i] - y_current_host[i]).dot((y_next_host[i] - y_current_host[i]));
				}
				delta_sum = delta_sum / particle_num;
				delta_sum = sqrt(delta_sum);

				y_current_host.release();
				y_next_host.release();

				if (delta_sum <= this->var_convergencyEpsilonField.getValue()) { // converge
					iterCount++;
					break;
				}
			}

			Function1Pt::copy(y_current, y_next);

			printf("Timestep: %f \n", this->getParent()->getDt());

			iterCount++;
		}

		printf("Iterations: %d \n", iterCount);
		if (this->var_isConvergeComputeField.getValue()) {
			if (this->iteration_curve.size() > this->iteration_curve_max_size) {
				printf("It's %d frame in curve now\n", this->current_frame_num);
				this->var_isConvergeComputeField.setValue(false);
				std::ofstream curve_output_file("../../../iteration_curve.csv");
				std::map<int, int>::iterator ite;
				for (ite = this->iteration_curve.begin(); ite != this->iteration_curve.end(); ++ite) {
					curve_output_file << ite->first << " , " << ite->second << std::endl;
				}
			}

			this->iteration_curve.insert(std::make_pair(this->current_frame_num, iterCount));
			this->current_frame_num++;
		}

		test_HM_UpdatePosition << <pDims, BLOCK_SIZE >> > (
			this->inPosition()->getValue(),
			this->inVelocity()->getValue(),
			y_next,
			m_position_old,
			this->getParent()->getDt());
		cuSynchronize();
	}


	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::solveElasticityGradientDescent()
	{

	}


	template <typename Real, typename Coord, typename Matrix>
	__global__ void HM_ComputeEnergy(
		GArray<Real> energy,
		GArray<Coord> eigens,
		EnergyType type)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= energy.size()) return;

#ifdef STVK_MODEL
		StVKModel<Real, Matrix>  model;
#endif
#ifdef LINEAR_MODEL
		LinearModel<Real, Matrix>  model;
#endif
#ifdef NEOHOOKEAN_MODEL
		NeoHookeanModel<Real, Matrix>  model;
#endif
#ifdef XU_MODEL
		XuModel<Real, Matrix>  model;
#endif

		Coord eigen_i = eigens[pId];

		energy[pId] = model.getEnergy(eigen_i[0], eigen_i[1], eigen_i[2]);
	}

	template<typename TDataType>
	void HyperelasticityModule_test<TDataType>::getEnergy(Real& totalEnergy, GArray<Coord>& position)
	{
		int numOfParticles = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		cuExecute(m_F.size(),
			HM_ComputeF,
			m_F,
			m_eigenValues,
			m_invK,
			m_validOfK,
			m_matU,
			m_matV,
			this->inRotation()->getValue(),
			position,
			this->inRestShape()->getValue(),
			this->inHorizon()->getValue());

		cuExecute(m_energy.size(),
			HM_Compute1DEnergy,
			m_energy,
			position,
			m_F,
			this->inVolume()->getValue(),
			m_validOfK,
			m_eigenValues,
			this->inRestShape()->getValue(),
			m_energyType);

// 		HM_ComputeEnergy <Real, Coord, Matrix> << <pDims, BLOCK_SIZE >> > (
// 			m_energy,
// 			m_eigenValues,
// 			m_energyType);
// 		cuSynchronize();

		totalEnergy = m_reduce->accumulate(m_energy.begin(), m_energy.size());
	}


#ifdef PRECISION_FLOAT
	template class HyperelasticityModule_test<DataType3f>;
#else
	template class HyperelasticityModule_test<DataType3d>;
#endif
}