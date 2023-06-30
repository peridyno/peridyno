#include "SemiImplicitHyperelasticitySolver.h"

#include "Matrix/MatrixFunc.h"
#include "ParticleSystem/Module/Kernel.h"
#include "curand_kernel.h"
#include "Algorithm/CudaRand.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SemiImplicitHyperelasticitySolver, TDataType);

	__constant__ EnergyModels<Real> ENERGY_FUNC;

	template<typename Real>
	__device__ Real constantWeight(Real r, Real h)
	{
		Real d = h / r;
		return d * d;
	}

	template<typename Real>
	__device__ Real D_Weight(Real r, Real h)
	{
		CorrectedKernel<Real> kernSmooth;
		return kernSmooth.WeightRR(r, 4 * h);
	}

	template<typename TDataType>
	SemiImplicitHyperelasticitySolver<TDataType>::SemiImplicitHyperelasticitySolver()
		: LinearElasticitySolver<TDataType>()
	{
		this->varIterationNumber()->setValue(5);
	}

	template<typename TDataType>
	SemiImplicitHyperelasticitySolver<TDataType>::~SemiImplicitHyperelasticitySolver()
	{
		mWeights.clear();
		mDisplacement.clear();
		mInvK.clear();
		mF.clear();
		mPosBuf.clear();
	}

	template <typename Real, typename Coord, typename Matrix>
	__global__ void HM_ComputeEnergy(
		DArray<Real> energy,
		DArray<Coord> eigens,
		EnergyType type)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= energy.size()) return;

		Coord eigen_i = eigens[pId];

		if (type == StVK) {
			energy[pId] = ENERGY_FUNC.stvkModel.getEnergy(eigen_i[0], eigen_i[1], eigen_i[2]);
		}
		else if (type == NeoHooekean) {
			energy[pId] = ENERGY_FUNC.neohookeanModel.getEnergy(eigen_i[0], eigen_i[1], eigen_i[2]);
		}
		else if (type == Linear) {
			energy[pId] = ENERGY_FUNC.linearModel.getEnergy(eigen_i[0], eigen_i[1], eigen_i[2]);
		}
		else if (type == Xuetal) {
			energy[pId] = ENERGY_FUNC.xuModel.getEnergy(eigen_i[0], eigen_i[1], eigen_i[2]);
		}
		else if (type == MooneyRivlin) {
			energy[pId] = ENERGY_FUNC.mrModel.getEnergy(eigen_i[0], eigen_i[1], eigen_i[2]);
		}
		else if (type == Fung) {
			energy[pId] = ENERGY_FUNC.fungModel.getEnergy(eigen_i[0], eigen_i[1], eigen_i[2]);
		}
		else if (type == Ogden) {
			energy[pId] = ENERGY_FUNC.ogdenModel.getEnergy(eigen_i[0], eigen_i[1], eigen_i[2]);
		}
		else if (type == Yeoh) {
			energy[pId] = ENERGY_FUNC.yeohModel.getEnergy(eigen_i[0], eigen_i[1], eigen_i[2]);
		}
		else if (type == ArrudaBoyce) {
			energy[pId] = ENERGY_FUNC.abModel.getEnergy(eigen_i[0], eigen_i[1], eigen_i[2]);
		}
	}


	template<typename TDataType>
	void SemiImplicitHyperelasticitySolver<TDataType>::solveElasticity()
	{
		cudaMemcpyToSymbol(ENERGY_FUNC, &this->inEnergyModels()->getData(), sizeof(EnergyModels<Real>));

		enforceHyperelasticity();
	}

	template <typename Coord>
	__global__ void HM_ComputeGradient(
		DArray<Coord> grad,
		DArray<Coord> y_pre,
		DArray<Coord> y_next)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		grad[pId] = y_next[pId] - y_pre[pId];
	}

	template <typename Real, typename Coord>
	__global__ void HM_ComputeCurrentPosition(
		DArray<Coord> grad,
		DArray<Coord> y_current,
		DArray<Coord> y_next,
		DArray<Attribute> attribute,
		DArray<Real> alpha)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		y_next[pId] = attribute[pId].isDynamic() ? y_current[pId] + alpha[pId] * grad[pId] : y_current[pId];
	}

	template <typename Real, typename Coord>
	__global__ void HM_ComputeCurrentPosition(
		DArray<Coord> grad,
		DArray<Coord> y_current,
		DArray<Coord> y_next,
		Real alpha)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		y_next[pId] = y_current[pId] + alpha * grad[pId];
	}

	template <typename Real, typename Coord, typename Matrix, typename Bond>
	__global__ void HM_Compute1DEnergy(
		DArray<Real> energy,
		DArray<Coord> energyGradient,
		DArray<Coord> X,
		DArray<Coord> Y,
		DArray<Matrix> F,
		DArray<Real> volume,
		DArray<bool> validOfK,
		DArray<Coord> eigenValues,
		DArrayList<Bond> bonds,
		EnergyType type)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= energy.size()) return;

		Coord y_i = Y[pId];

		Real totalEnergy = 0.0f;
		Coord totalEnergyGradient = Coord(0);
		Real V_i = volume[pId];

		int size_i = bonds[pId].size();

		Coord x_i = X[pId];
		Coord eigen_value_i = eigenValues[pId];
		bool valid_i = validOfK[pId];

		Matrix F_i = F[pId];

		for (int ne = 1; ne < size_i; ne++)
		{
			Bond bond_ij = bonds[pId][ne];
			int j = bond_ij.idx;
			Coord y_j = Y[j];
			Real r = bond_ij.xi.norm();

			Real V_j = volume[j];

			if (r > EPSILON)
			{
				Real norm_ij = (y_j - y_i).norm();
				Real lambda_ij = norm_ij / r;

				Real deltaEnergy;
				Coord deltaEnergyGradient;
				Coord dir_ij = norm_ij < EPSILON ? Coord(0) : (y_i - y_j) / (r);

				if (type == StVK) {
					deltaEnergy = V_j * ENERGY_FUNC.stvkModel.getEnergy(lambda_ij, lambda_ij, lambda_ij);
					deltaEnergyGradient = V_j * (ENERGY_FUNC.stvkModel.getStressTensorPositive(lambda_ij, lambda_ij, lambda_ij) - ENERGY_FUNC.stvkModel.getStressTensorNegative(lambda_ij, lambda_ij, lambda_ij)) * dir_ij;
				}
				else if (type == NeoHooekean) {
					deltaEnergy = V_j * ENERGY_FUNC.neohookeanModel.getEnergy(lambda_ij, lambda_ij, lambda_ij);
					deltaEnergyGradient = V_j * (ENERGY_FUNC.neohookeanModel.getStressTensorPositive(lambda_ij, lambda_ij, lambda_ij) - ENERGY_FUNC.neohookeanModel.getStressTensorNegative(lambda_ij, lambda_ij, lambda_ij)) * dir_ij;
				}
				else if (type == Linear) {
					deltaEnergy = V_j * ENERGY_FUNC.linearModel.getEnergy(lambda_ij, lambda_ij, lambda_ij);
					deltaEnergyGradient = V_j * (ENERGY_FUNC.linearModel.getStressTensorPositive(lambda_ij, lambda_ij, lambda_ij) - ENERGY_FUNC.linearModel.getStressTensorNegative(lambda_ij, lambda_ij, lambda_ij)) * dir_ij;
				}
				else if (type == Xuetal) {
					deltaEnergy = V_j * ENERGY_FUNC.xuModel.getEnergy(lambda_ij, lambda_ij, lambda_ij);
					deltaEnergyGradient = V_j * (ENERGY_FUNC.xuModel.getStressTensorPositive(lambda_ij, lambda_ij, lambda_ij) - ENERGY_FUNC.xuModel.getStressTensorNegative(lambda_ij, lambda_ij, lambda_ij)) * dir_ij;
				}
				else if (type == MooneyRivlin) {
					deltaEnergy = V_j * ENERGY_FUNC.mrModel.getEnergy(lambda_ij, lambda_ij, lambda_ij);
					deltaEnergyGradient = V_j * (ENERGY_FUNC.mrModel.getStressTensorPositive(lambda_ij, lambda_ij, lambda_ij) - ENERGY_FUNC.xuModel.getStressTensorNegative(lambda_ij, lambda_ij, lambda_ij)) * dir_ij;
				}
				else if (type == Fung) {
					deltaEnergy = V_j * ENERGY_FUNC.fungModel.getEnergy(lambda_ij, lambda_ij, lambda_ij);
					deltaEnergyGradient = V_j * (ENERGY_FUNC.fungModel.getStressTensorPositive(lambda_ij, lambda_ij, lambda_ij) - ENERGY_FUNC.xuModel.getStressTensorNegative(lambda_ij, lambda_ij, lambda_ij)) * dir_ij;
				}
				else if (type == Ogden) {
					deltaEnergy = V_j * ENERGY_FUNC.ogdenModel.getEnergy(lambda_ij, lambda_ij, lambda_ij);
					deltaEnergyGradient = V_j * (ENERGY_FUNC.ogdenModel.getStressTensorPositive(lambda_ij, lambda_ij, lambda_ij) - ENERGY_FUNC.xuModel.getStressTensorNegative(lambda_ij, lambda_ij, lambda_ij)) * dir_ij;
				}
				else if (type == Yeoh) {
					deltaEnergy = V_j * ENERGY_FUNC.yeohModel.getEnergy(lambda_ij, lambda_ij, lambda_ij);
					deltaEnergyGradient = V_j * (ENERGY_FUNC.yeohModel.getStressTensorPositive(lambda_ij, lambda_ij, lambda_ij) - ENERGY_FUNC.xuModel.getStressTensorNegative(lambda_ij, lambda_ij, lambda_ij)) * dir_ij;
				}
				else if (type == ArrudaBoyce) {
					deltaEnergy = V_j * ENERGY_FUNC.abModel.getEnergy(lambda_ij, lambda_ij, lambda_ij);
					deltaEnergyGradient = V_j * (ENERGY_FUNC.abModel.getStressTensorPositive(lambda_ij, lambda_ij, lambda_ij) - ENERGY_FUNC.xuModel.getStressTensorNegative(lambda_ij, lambda_ij, lambda_ij)) * dir_ij;
				}

				totalEnergy += deltaEnergy;
				totalEnergyGradient += deltaEnergyGradient;
			}
		}

		energy[pId] = totalEnergy * V_i;
		energyGradient[pId] = totalEnergyGradient * V_i;
	}

	template <typename Coord>
	__global__ void HM_Chebyshev_Acceleration(DArray<Coord> next_X, DArray<Coord> X, DArray<Coord> prev_X, float omega)
	{
		int pId = blockDim.x * blockIdx.x + threadIdx.x;
		if (pId >= prev_X.size())	return;

		next_X[pId] = (next_X[pId] - X[pId]) * 0.666 + X[pId];

		next_X[pId] = omega * (next_X[pId] - prev_X[pId]) + prev_X[pId];
	}

	template <typename Real, typename Coord, typename Matrix, typename Bond>
	__global__ void HM_ComputeStepLength(
		DArray<Real> stepLength,
		DArray<Coord> gradient,
		DArray<Coord> energyGradient,
		DArray<Real> volume,
		DArray<Matrix> A,
		DArray<Real> energy,
		DArrayList<Bond> restShapes)
	{
		int pId = blockDim.x * blockIdx.x + threadIdx.x;
		if (pId >= stepLength.size())	return;

		//TODO: replace 1000 with an input
		Real mass_i = volume[pId] * 1000.0;
		Real energy_i = energy[pId];

		Real deltaE_i = abs(energyGradient[pId].dot(gradient[pId]));

		Real alpha = deltaE_i < EPSILON || deltaE_i < energy_i ? Real(1) : energy_i / deltaE_i;

		alpha /= Real(1 + restShapes[pId].size());

		stepLength[pId] = alpha;
	}

	template <typename Coord>
	__global__ void HM_FixCoM(
		DArray<Coord> positions,
		DArray<Attribute> atts,
		Coord CoM)
	{
		int pId = blockDim.x * blockIdx.x + threadIdx.x;
		if (pId >= positions.size()) return;
		if ((positions[pId] - CoM).norm() < 0.05)
			atts[pId].setFixed();
	}

	template <typename Coord, typename Bond>
	__global__ void K_UpdateRestShape(
		DArrayList<Bond> shape,
		DArrayList<int> nbr,
		DArray<Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		Bond np;

		List<Bond>& rest_shape_i = shape[pId];
		List<int>& list_id_i = nbr[pId];
		int nbSize = list_id_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_id_i[ne];
			np.index = j;
			np.pos = pos[j];
			np.weight = 1;

			rest_shape_i.insert(np);
			if (pId == j)
			{
				Bond np_0 = rest_shape_i[0];
				rest_shape_i[0] = np;
				rest_shape_i[ne] = np_0;
			}
		}
	}

	template <typename Coord>
	__global__ void test_HM_UpdatePosition(
		DArray<Coord> position,
		DArray<Coord> velocity,
		DArray<Coord> y_next,
		DArray<Coord> position_old,
		DArray<Attribute> att,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		if (att[pId].isDynamic()) {
			position[pId] = y_next[pId];
			velocity[pId] += (position[pId] - position_old[pId]) / dt;
		}
	}

	template<typename Real>
	__device__ Real LimitStrain(Real s, Real slimit)
	{
		Real ret;
		// 		if (s < 0)
		// 		{
		// 			ret = 0;// s + 0.05 * (slimit - s);
		// 		}
		// // 		else if (s > 1.1 / slimit)
		// // 		{
		// // 			ret = s - 0.05 * (s - 1 / slimit);
		// // 		}
		// 		else
		{
			ret = clamp(s, Real(0.2), 1 / slimit);
		}

		return ret;
	}

	template <typename Real, typename Coord, typename Matrix, typename Bond>
	__global__ void HM_ComputeF(
		DArray<Matrix> F,
		DArray<Coord> eigens,
		DArray<Matrix> invK,
		DArray<bool> validOfK,
		DArray<Matrix> matU,
		DArray<Matrix> matV,
		DArray<Matrix> Rots,
		DArray<Coord> X,
		DArray<Coord> Y,
		DArrayList<Bond> bonds,
		Real strainLimiting,
		Real horizon)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Y.size()) return;
		/*if (pId % 100 == 0)
			printf("%d %d \n", pId, pId);*/

		Coord x_i = X[pId];

		Real total_weight = Real(0);
		Matrix matL_i(0);
		Matrix matK_i(0);

#ifdef DEBUG_INFO
		if (pId == 497)
		{
			printf("Position in HM_ComputeF %d: %f %f %f \n", pId, Y[pId][0], Y[pId][1], Y[pId][2]);
		}
#endif // DEBUG_INFO
		
		List<Bond>& bonds_i = bonds[pId];

		Real maxDist = Real(0);
		for (int ne = 0; ne < bonds_i.size(); ne++)
		{
			maxDist = max(maxDist, bonds_i[ne].xi.norm());
		}
		maxDist = maxDist < EPSILON ? Real(1) : maxDist;

		//printf("maxdist %d: %f; \n\n", pId, maxDist);

#ifdef DEBUG_INFO
		printf("Max distance %d: %f \n", pId, maxDist);
#endif // DEBUG_INFO

		for (int ne = 0; ne < bonds_i.size(); ne++)
		{
			Bond bond_ij = bonds[pId][ne];
			int j = bond_ij.idx;
			Coord x_j = X[j];
			Real r = (x_i - x_j).norm();

			if (r > EPSILON)
			{
				Coord p = (Y[j] - Y[pId]) / maxDist;
				Coord q = (x_j - x_i) / maxDist;
				//Real weight = D_Weight(r, horizon);
				Real weight = 1.;

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
					printf("%d Neighbor %d: %f %f %f \n", pId, j, Y[j][0], Y[j][1], Y[j][2]);
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

			mat_out = U * D * V.transpose();
			printf("matK polar: \n %f %f %f \n %f %f %f \n %f %f %f \n\n",
				mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
				mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
				mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

			printf("Horizon: %f; Det: %f \n", horizon, matK_i.determinant());
		}
#endif // DEBUG_INFO

		Real maxK = maximum(abs(D(0, 0)), maximum(abs(D(1, 1)), abs(D(2, 2))));
		Real minK = minimum(abs(D(0, 0)), minimum(abs(D(1, 1)), abs(D(2, 2))));

		bool valid_K = (minK < EPSILON) ? false : true;

		validOfK[pId] = valid_K;

		Matrix F_i;
		if (valid_K)
		{
			invK[pId] = matK_i.inverse();
			F_i = matL_i * matK_i.inverse();
		}
		else
		{
			Real threshold = 0.0001f * horizon;
			D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
			D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
			D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;
			invK[pId] = V * D * U.transpose();
			F_i = matL_i * V * D * U.transpose();
		}

		polarDecomposition(F_i, R, U, D, V);

		// 		Real minDist = Real(1000.0f);
		// 		for (int ne = 0; ne < size_i; ne++)
		// 		{
		// 			Bond np_j = restShapes[pId][ne];
		// 			int j = np_j.index;
		// 			Coord rest_pos_j = np_j.pos;
		// 			Real r = (rest_pos_i - rest_pos_j).norm();
		// 
		// 			if(r > EPSILON)
		// 				minDist = min(minDist, r);
		// 		}
		// 
		// 		printf("ID %d: min value %f \n", pId, minDist);

		if (F_i.determinant() < maxDist * EPSILON)
		{
		}
		else
		{
			matU[pId] = U;
			matV[pId] = V;
		}

		// 		Coord n0 = Coord(1, 0, 0);
		// 		Coord n1 = Coord(0, 1, 0);
		// 		Coord n2 = Coord(0, 0, 1);
		// 
		// 		Coord n0_rot = R * n0;
		// 		Coord n1_rot = R * n1;
		// 		Coord n2_rot = R * n2;

		Real l0 = D(0, 0);
		Real l1 = D(1, 1);
		Real l2 = D(2, 2);

		const Real slimit = strainLimiting;

		l0 = clamp(l0, slimit, 1 / slimit);
		l1 = clamp(l1, slimit, 1 / slimit);
		l2 = clamp(l2, slimit, 1 / slimit);

		D(0, 0) = l0;
		D(1, 1) = l1;
		D(2, 2) = l2;

		eigens[pId] = Coord(l0, l1, l2);

		F[pId] = U * D * V.transpose();

		// 		if (F_i.determinant() < maxDist*EPSILON)
		// 		{
		// 			R = Rots[pId];
		// 		}
		// 		else
		// 		{
		// 			Rots[pId] = R;
		// 		}
		// 			
		// 		Coord n0 = Coord(1, 0, 0);
		// 		Coord n1 = Coord(0, 1, 0);
		// 		Coord n2 = Coord(0, 0, 1);
		// 
		// 		Coord n0_rot = R * n0;
		// 		Coord n1_rot = R * n1;
		// 		Coord n2_rot = R * n2;
		// 
		// 		Real l0 = n0_rot.dot(F_i*n0);
		// 		Real l1 = n1_rot.dot(F_i*n1);
		// 		Real l2 = n2_rot.dot(F_i*n2);
		// 
		// 		matU[pId] = R;
		// 		matV[pId] = Matrix::identityMatrix();
		// 
		// 		const Real slimit = strainLimiting;
		// 
		// 		l0 = clamp(l0, slimit, 1 / slimit);
		// 		l1 = clamp(l1, slimit, 1 / slimit);
		// 		l2 = clamp(l2, slimit, 1 / slimit);
		// 
		// 		D(0, 0) = l0;
		// 		D(1, 1) = l1;
		// 		D(2, 2) = l2;
		// 
		// 		eigens[pId] = Coord(l0, l1, l2);
		// 
		// 		F[pId] = R * D;
	}

	template <typename Real, typename Coord, typename Matrix, typename Bond>
	__global__ void HM_JacobiStepNonsymmetric(
		DArray<Coord> source,
		DArray<Matrix> A,
		DArray<Coord> y_pre,
		DArray<Matrix> matU,
		DArray<Matrix> matV,
		DArray<Matrix> matR,
		DArray<Coord> eigen,
		DArray<bool> validOfK,
		DArray<Matrix> F,
		DArray<Coord> X,
		DArrayList<Bond> bonds,
		Real horizon,
		DArray<Real> volume,
		DArrayList<Real> volumePair,
		Real dt,
		EnergyType type)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_pre.size()) return;

		Coord x_i = X[pId];
		int size_i = bonds[pId].size();

		Real kappa = 4 / (3 * M_PI * horizon * horizon * horizon);
		Real lambda_i1 = eigen[pId][0];
		Real lambda_i2 = eigen[pId][1];
		Real lambda_i3 = eigen[pId][2];

		Matrix U_i = matU[pId];
		Matrix V_i = matV[pId];

		Matrix S1_i;
		Matrix S2_i;
		if (type == StVK) {
			S1_i = ENERGY_FUNC.stvkModel.getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3);
			S2_i = ENERGY_FUNC.stvkModel.getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3);
		}
		else if (type == NeoHooekean) {
			S1_i = ENERGY_FUNC.neohookeanModel.getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3);
			S2_i = ENERGY_FUNC.neohookeanModel.getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3);
		}
		else if (type == Linear) {
			S1_i = ENERGY_FUNC.linearModel.getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3);
			S2_i = ENERGY_FUNC.linearModel.getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3);
		}
		else if (type == Xuetal) {
			S1_i = ENERGY_FUNC.xuModel.getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3);
			S2_i = ENERGY_FUNC.xuModel.getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3);
		}
		else if (type == MooneyRivlin) {
			S1_i = ENERGY_FUNC.mrModel.getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3);
			S2_i = ENERGY_FUNC.mrModel.getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3);
		}
		else if (type == Fung) {
			S1_i = ENERGY_FUNC.fungModel.getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3);
			S2_i = ENERGY_FUNC.fungModel.getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3);
		}
		else if (type == Ogden) {
			S1_i = ENERGY_FUNC.ogdenModel.getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3);
			S2_i = ENERGY_FUNC.ogdenModel.getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3);
		}
		else if (type == Yeoh) {
			S1_i = ENERGY_FUNC.yeohModel.getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3);
			S2_i = ENERGY_FUNC.yeohModel.getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3);
		}
		else if (type == ArrudaBoyce) {
			S1_i = ENERGY_FUNC.abModel.getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3);
			S2_i = ENERGY_FUNC.abModel.getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3);
		}

		Matrix PK1_i = U_i * S1_i * U_i.transpose();
		Matrix PK2_i = U_i * S2_i * V_i.transpose();

		//Real Vol_i = volume[pId];

		Matrix F_i = F[pId];

		Coord y_pre_i = y_pre[pId];

		bool K_valid = validOfK[pId];

		Matrix mat_i(0);
		Coord source_i(0);

		for (int ne = 0; ne < size_i; ne++)
		{
			Bond bond_ij = bonds[pId][ne];
			int j = bond_ij.idx;
			Coord x_j = X[j];
			Coord y_pre_j = y_pre[j];
			Real r = bond_ij.xi.norm();

			if (r > EPSILON)
			{
				Real weight_ij = 1.;
				//Real Vol_j = volume[j];
				Real V_ij = volumePair[pId][ne];

				Coord y_ij = y_pre_i - y_pre_j;

				Real lambda = y_ij.norm() / r;

				//const Real scale = Vol_i * Vol_j*kappa;
				const Real scale = V_ij * V_ij * kappa;
				const Real sw_ij = dt * dt * scale * weight_ij / (r * r);

				bool K_valid_ij = K_valid & validOfK[j];

				Matrix S1_ij;
				Matrix S2_ij;
				if (type == StVK) {
					S1_ij = ENERGY_FUNC.stvkModel.getStressTensorPositive(lambda, lambda, lambda);
					S2_ij = ENERGY_FUNC.stvkModel.getStressTensorNegative(lambda, lambda, lambda);
				}
				else if (type == NeoHooekean) {
					S1_ij = ENERGY_FUNC.neohookeanModel.getStressTensorPositive(lambda, lambda, lambda);
					S2_ij = ENERGY_FUNC.neohookeanModel.getStressTensorNegative(lambda, lambda, lambda);
				}
				else if (type == Linear) {
					S1_ij = ENERGY_FUNC.linearModel.getStressTensorPositive(lambda, lambda, lambda);
					S2_ij = ENERGY_FUNC.linearModel.getStressTensorNegative(lambda, lambda, lambda);
				}
				else if (type == Xuetal) {
					S1_ij = ENERGY_FUNC.xuModel.getStressTensorPositive(lambda, lambda, lambda);
					S2_ij = ENERGY_FUNC.xuModel.getStressTensorNegative(lambda, lambda, lambda);
				}
				else if (type == MooneyRivlin) {
					S1_ij = ENERGY_FUNC.mrModel.getStressTensorPositive(lambda, lambda, lambda);
					S2_ij = ENERGY_FUNC.mrModel.getStressTensorNegative(lambda, lambda, lambda);
				}
				else if (type == Fung) {
					S1_ij = ENERGY_FUNC.fungModel.getStressTensorPositive(lambda, lambda, lambda);
					S2_ij = ENERGY_FUNC.fungModel.getStressTensorNegative(lambda, lambda, lambda);
				}
				else if (type == Ogden) {
					S1_ij = ENERGY_FUNC.ogdenModel.getStressTensorPositive(lambda, lambda, lambda);
					S2_ij = ENERGY_FUNC.ogdenModel.getStressTensorNegative(lambda, lambda, lambda);
				}
				else if (type == Yeoh) {
					S1_ij = ENERGY_FUNC.yeohModel.getStressTensorPositive(lambda, lambda, lambda);
					S2_ij = ENERGY_FUNC.yeohModel.getStressTensorNegative(lambda, lambda, lambda);
				}
				else if (type == ArrudaBoyce) {
					S1_ij = ENERGY_FUNC.abModel.getStressTensorPositive(lambda, lambda, lambda);
					S2_ij = ENERGY_FUNC.abModel.getStressTensorNegative(lambda, lambda, lambda);
				}

				Matrix PK1_ij = K_valid_ij ? PK1_i : S1_ij;
				Matrix PK2_ij = K_valid_ij ? PK2_i : S2_ij;

				Coord dir_ij = lambda > EPSILON ? y_ij.normalize() : Coord(1, 0, 0);

				Coord x_ij = K_valid_ij ? x_i - x_j : dir_ij * (x_i - x_j).norm();

				Coord ds_ij = sw_ij * PK1_ij * y_pre_j + sw_ij * PK2_ij * x_ij;
				Coord ds_ji = sw_ij * PK1_ij * y_pre_i - sw_ij * PK2_ij * x_ij;

				Matrix mat_ij = sw_ij * PK1_ij;

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

	template <typename Real, typename Coord, typename Matrix>
	__global__ void HM_ComputeNextPosition(
		DArray<Coord> y_next,
		DArray<Coord> y_pre,
		DArray<Coord> y_old,
		DArray<Coord> source,
		DArray<Matrix> A,
		DArray<Real> volume,
		DArray<Attribute> attribute,
		EnergyType type)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		//TODO: replace 1000 with an input
		Real mass_i = volume[pId] * 1000;

		Matrix mat_i = A[pId] + mass_i * Matrix::identityMatrix();
		Coord src_i = source[pId] + mass_i * y_old[pId];

		y_next[pId] = attribute[pId].isDynamic() ? mat_i.inverse() * src_i : y_pre[pId];
	}

	template <typename Matrix>
	__global__ void HM_InitRotation(
		DArray<Matrix> U,
		DArray<Matrix> V,
		DArray<Matrix> R)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= U.size()) return;

		U[pId] = Matrix::identityMatrix();
		V[pId] = Matrix::identityMatrix();
		R[pId] = Matrix::identityMatrix();
	}

	template<typename TDataType>
	void SemiImplicitHyperelasticitySolver<TDataType>::resizeAllFields()
	{
		uint num = this->inY()->size();

		if (m_F.size() == num)
			return;

		m_F.resize(num);
		m_invF.resize(num);
		m_invK.resize(num);
		m_validOfK.resize(num);
		m_validOfF.resize(num);

		m_eigenValues.resize(num);
		m_matU.resize(num);
		m_matV.resize(num);
		m_matR.resize(num);

		m_energy.resize(num);
		m_alpha.resize(num);
		m_gradient.resize(num);
		mEnergyGradient.resize(num);

		y_next.resize(num);
		y_pre.resize(num);
		y_current.resize(num);
		y_residual.resize(num);
		y_gradC.resize(num);

		m_source.resize(num);
		m_A.resize(num);

		mPosBuf.resize(num);

		m_fraction.resize(num);

		cuExecute(m_matU.size(),
			HM_InitRotation,
			m_matU,
			m_matV,
			m_matR);

		m_reduce = Reduction<Real>::Create(num);
	}

	template<typename TDataType>
	void SemiImplicitHyperelasticitySolver<TDataType>::enforceHyperelasticity()
	{
		//std::cout << "enforceElasticity\n";

		resizeAllFields();

		int numOfParticles = this->inY()->size();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		std::cout << "enforceElasticity " << numOfParticles << std::endl;

		this->mWeights.reset();

		Log::sendMessage(Log::User, "\n \n \n \n *************solver start!!!***************");

		/**************************** Jacobi method ************************************************/
		// initialize y_now, y_next_iter
		y_current.assign(this->inY()->getData());
		y_next.assign(this->inY()->getData());
		mPosBuf.assign(this->inY()->getData());

		// do Jacobi method Loop
		bool convergeFlag = false; // converge or not
		int iterCount = 0;

		Real omega;
		Real alpha = 1.0f;
		std::cout << "enforceElasticity 2  " << numOfParticles << std::endl;

		while (iterCount < this->varIterationNumber()->getData()) {
			//printf("%.3lf\n", inHorizon()->getData());

			
			m_source.reset();
			m_A.reset();
			HM_ComputeF << <pDims, BLOCK_SIZE >> > (
				m_F,
				m_eigenValues,
				m_invK,
				m_validOfK,
				m_matU,
				m_matV,
				m_matR,
				this->inX()->getData(),
				y_current,
				this->inBonds()->getData(),
				this->varStrainLimiting()->getData(),
				this->inHorizon()->getData());
			cuSynchronize();
			
			HM_JacobiStepNonsymmetric << <pDims, BLOCK_SIZE >> > (
				m_source,
				m_A,
				y_current,
				m_matU,
				m_matV,
				m_matR,
				m_eigenValues,
				m_validOfK,
				m_F,
				this->inX()->getData(),
				this->inBonds()->getData(),
				this->inHorizon()->getData(),
				this->inVolume()->getData(),
				this->inVolumePair()->getData(),
				this->inTimeStep()->getData(),
				this->inEnergyType()->getData());
			cuSynchronize();
			
			cuExecute(y_next.size(),
				HM_ComputeNextPosition,
				y_next,
				y_current,
				mPosBuf,
				m_source,
				m_A,
				this->inVolume()->getData(),
				this->inAttribute()->getData(),
				this->inEnergyType()->getData());
			
			//sub steping
			cuExecute(m_gradient.size(),
				HM_ComputeGradient,
				m_gradient,
				y_current,
				y_next);
			

			cuExecute(m_energy.size(),
				HM_Compute1DEnergy,
				m_energy,
				mEnergyGradient,
				this->inX()->getData(),
				y_current,
				m_F,
				this->inVolume()->getData(),
				m_validOfK,
				m_eigenValues,
				this->inBonds()->getData(),
				this->inEnergyType()->getData());
			
			m_alphaCompute = this->varIsAlphaComputed()->getData();
			if (this->m_alphaCompute) {
				cuExecute(m_alpha.size(),
					HM_ComputeStepLength,
					m_alpha,
					m_gradient,
					mEnergyGradient,
					this->inVolume()->getData(),
					m_A,
					m_energy,
					this->inBonds()->getData());

				cuExecute(m_gradient.size(),
					HM_ComputeCurrentPosition,
					m_gradient,
					y_current,
					y_next,
					this->inAttribute()->getData(),
					m_alpha);
				
			}
			else {
				alpha = Real(0.15f);
				cuExecute(m_gradient.size(),
					HM_ComputeCurrentPosition,
					m_gradient,
					y_current,
					y_next,
					alpha);
			}

			y_current.assign(y_next);

			iterCount++;
		}

		cuExecute(this->inY()->getDataPtr()->size(),
			test_HM_UpdatePosition,
			this->inY()->getData(),
			this->inVelocity()->getData(),
			y_next,
			mPosBuf,
			this->inAttribute()->getData(),
			this->inTimeStep()->getData());
		printf("outside\n");
	}

	DEFINE_CLASS(SemiImplicitHyperelasticitySolver);
}