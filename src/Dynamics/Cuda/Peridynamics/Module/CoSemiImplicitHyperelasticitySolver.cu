#include "CoSemiImplicitHyperelasticitySolver.h"
#include "Matrix/MatrixFunc.h"
#include "ParticleSystem/Module/Kernel.h"
#include "curand_kernel.h"
#include "Algorithm/CudaRand.h"

namespace dyno
{
	IMPLEMENT_TCLASS(CoSemiImplicitHyperelasticitySolver, TDataType);

	__constant__ EnergyModels<Real> ENERGY_FUNC;

	template<typename Real>
	__device__ Real constantWeight(Real r, Real h)
	{
		Real d = h / r;
		return d * d;
	}

	template<typename TDataType>
	CoSemiImplicitHyperelasticitySolver<TDataType>::CoSemiImplicitHyperelasticitySolver()
		: LinearElasticitySolver<TDataType>()
	{
		
		mContactRule = std::make_shared<ContactRule<TDataType>>();
		this->varIterationNumber()->setValue(30);
		
	}

	template<typename TDataType>
	void CoSemiImplicitHyperelasticitySolver<TDataType>::connectContact() {
		
	}

	template<typename TDataType>
	void CoSemiImplicitHyperelasticitySolver<TDataType>::initializeVolume()
	{
		int numOfParticles = this->inY()->getData().size();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);
		std::cout << "dev: " << numOfParticles << " particles\n";
		HM_InitVolume << <pDims, BLOCK_SIZE >> > (m_volume, m_objectVolume, m_objectVolumeSet, m_particleVolume, m_particleVolumeSet);
	}

	template<typename TDataType>
	CoSemiImplicitHyperelasticitySolver<TDataType>::~CoSemiImplicitHyperelasticitySolver()
	{
		mWeights.clear();
		mDisplacement.clear();
		mInvK.clear();
		mF.clear();
		mPosBuf.clear();
		mPosBuf_March.clear();
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
	}


	template<typename TDataType>
	void CoSemiImplicitHyperelasticitySolver<TDataType>::solveElasticity()
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
		DArray<Real> alpha)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		y_next[pId] = y_current[pId] + alpha[pId] * grad[pId];
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
		DArray<Coord> pos_current,
		DArray<Matrix> F,
		DArray<Real> volume,
		DArray<bool> validOfK,
		DArray<Coord> eigenValues,
		DArrayList<Bond> bonds,
		EnergyType type)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= energy.size()) return;

		Coord pos_current_i = pos_current[pId];

		Real totalEnergy = 0.0f;
		Coord totalEnergyGradient = Coord(0);
		Real V_i = volume[pId];

		int size_i = bonds[pId].size();

		Coord x_i = X[pId];
		Coord eigen_value_i = eigenValues[pId];
		bool valid_i = validOfK[pId];

		Matrix F_i = F[pId];

		for (int ne = 0; ne < size_i; ne++)
		{
			Bond bond_ij = bonds[pId][ne];
			int j = bond_ij.idx;
			Coord pos_current_j = pos_current[j];
			Coord x_j = X[j];
			Real r = (x_j - x_i).norm();

			Real V_j = volume[j];

			if (r > EPSILON)
			{
				Real norm_ij = (pos_current_j - pos_current_i).norm();
				Real lambda_ij = norm_ij / r;

				Real deltaEnergy;
				Coord deltaEnergyGradient;
				Coord dir_ij = norm_ij < EPSILON ? Coord(0) : (pos_current_i - pos_current_j) / (r);

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
		DArrayList<Bond> bonds)
	{
		int pId = blockDim.x * blockIdx.x + threadIdx.x;
		if (pId >= stepLength.size())	return;

		Real mass_i = volume[pId] * 1000.0;
		Real energy_i = energy[pId];

		Real deltaE_i = abs(energyGradient[pId].dot(gradient[pId]));

		Real alpha = deltaE_i < EPSILON || deltaE_i < energy_i ? Real(1) : energy_i / deltaE_i;

		alpha /= Real(1 + bonds[pId].size());

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
		DArray<Coord> position, //position target
		DArray<Coord> y_next, //position reference
		DArray<Attribute> att)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		if (!att[pId].isFixed()) {
			position[pId] = y_next[pId];
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

		if (!att[pId].isFixed()) {
			position[pId] = y_next[pId];

			velocity[pId] += (position[pId] - position_old[pId]) / dt;
		}
	}

	template <typename Coord>
	__global__ void test_HM_UpdateVelocity(
		DArray<Coord> position,
		DArray<Coord> velocity,
		DArray<Coord> v_even,
		DArray<Coord> y_next,
		DArray<Coord> position_old,
		DArray<Coord> contactForce,
		DArray<Coord> contactForce_target,
		DArray<Coord> forceTarget,
		DArray<Attribute> att,
		DArray<Coord> N,
		DArray<Coord> v_group,
		DArray<Real> weight,
		Real s,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		forceTarget[pId] = Coord(0.0);
		auto myNorm = [](Coord v1) {
			return sqrt(pow(v1[0], 2) + pow(v1[1], 2) + pow(v1[2], 2));
		};

		if (!att[pId].isFixed()) {
			contactForce_target[pId] = s * contactForce[pId];
			if (myNorm(contactForce_target[pId]) >= EPSILON) {
				Coord contactForce_mom = contactForce_target[pId] * dt;
				
				velocity[pId] -= v_even[0];
				Real alpha = -contactForce_mom.dot(velocity[pId]) / pow(myNorm(contactForce_mom), 2);
				//printf("alpha %f\n", alpha);
				//alpha =  minimum(alpha, (Real)10);
				//alpha = maximum(alpha, (Real)-10);
				velocity[pId] += alpha * contactForce_mom;
				velocity[pId] += v_even[0];
				contactForce_target[pId] *= alpha;
				
			}
		}
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
		Real horizon,
		Real const strainLimit,
		DArray<Coord> restNorm, 
		DArray<Coord> Norm)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Y.size()) return;

		Coord x_i = X[pId];
		int size_i = bonds[pId].size();
		Real total_weight = Real(0);
		Matrix matL_i(0);
		Matrix matK_i(0);

		Real t = 1;

#ifdef DEBUG_INFO
		if (pId == 497)
		{
			printf("Position in HM_ComputeF %d: %f %f %f \n", pId, Y[pId][0], Y[pId][1], Y[pId][2]);
		}
#endif // DEBUG_INFO
		//printf("%d %d \n", pId, size_i);

		Real maxDist = Real(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			Bond bond_ij = bonds[pId][ne];
			int j = bond_ij.idx;
			Coord y_j = X[j];
			Real r = (x_i - y_j).norm();

			maxDist = max(maxDist, r);
		}
		maxDist = maxDist < EPSILON ? Real(1) : maxDist;

#ifdef DEBUG_INFO
		printf("Max distance %d: %f \n", pId, maxDist);
#endif // DEBUG_INFO

		for (int ne = 0; ne < size_i; ne++)
		{
			Bond bond_ij = bonds[pId][ne];
			int j = bond_ij.idx;
			Coord x_j = X[j];
			Real r = (x_i - x_j).norm();

			if (r > EPSILON)
			{
				Real weight = Real(1);

				Coord p = (Y[j] - Y[pId]) / maxDist;
				Coord q = (x_j - x_i) / maxDist;
			

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
	

			Coord n = Norm[pId];
			Coord nr = restNorm[pId];
			matK_i(0, 0) += t*nr[0] * nr[0]; matK_i(0, 1) += t*nr[0] * nr[1]; matK_i(0, 2) += t*nr[0] * nr[2];
			matK_i(1, 0) += t*nr[1] * nr[0]; matK_i(1, 1) += t*nr[1] * nr[1]; matK_i(1, 2) += t*nr[1] * nr[2];
			matK_i(2, 0) += t*nr[2] * nr[0]; matK_i(2, 1) += t*nr[2] * nr[1]; matK_i(2, 2) += t*nr[2] * nr[2];
		

		
				matL_i(0, 0) += t*n[0] * nr[0]; matL_i(0, 1) += t*n[0] * nr[1]; matL_i(0, 2) += t*n[0] * nr[2];
				matL_i(1, 0) += t*n[1] * nr[0]; matL_i(1, 1) += t*n[1] * nr[1]; matL_i(1, 2) += t*n[1] * nr[2];
				matL_i(2, 0) += t*n[2] * nr[0]; matL_i(2, 1) += t*n[2] * nr[1]; matL_i(2, 2) += t*n[2] * nr[2];

				total_weight += 2 * t;
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

		Real maxK = maximum(abs(D(0, 0)), maximum(abs(D(1, 1)),abs(D(2,2))));
		Real minK = minimum(abs(D(0, 0)), minimum(abs(D(1, 1)),abs(D(2,2))));
		

		bool valid_K = (minK < EPSILON || maxK / minK > Real(1/(strainLimit * strainLimit))) ? false : true;
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
			F_i = matL_i * invK[pId];
		}

		polarDecomposition(F_i, R, U, D, V);

#ifdef DEBUG_INFO
		if (pId == 497)
		{
			Matrix mat_out = F_i;
			printf("matF_i: \n %f %f %f \n %f %f %f \n %f %f %f \n\n",
				mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
				mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
				mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

			mat_out = D;
			printf("matD: \n %f %f %f \n %f %f %f \n %f %f %f \n\n",
				mat_out(0, 0), mat_out(0, 1), mat_out(0, 2),
				mat_out(1, 0), mat_out(1, 1), mat_out(1, 2),
				mat_out(2, 0), mat_out(2, 1), mat_out(2, 2));

			printf("Horizon: %f; Det: %f \n", horizon, F_i.determinant());
		}
#endif // DEBUG_INFO
		if (F_i.determinant() < maxDist * EPSILON)
		{
			R = Rots[pId];
		}
		else
			Rots[pId] = R;

		//strain limit
		Coord n0 = Coord(1, 0, 0);
		Coord n1 = Coord(0, 1, 0);
		Coord n2 = Coord(0, 0, 1);

#ifdef DEBUG_INFO
		Coord n0_rot = R * n0;
		Coord n1_rot = R * n1;
		Coord n2_rot = R * n2;

		Real un0 = n0_rot.dot(n);
		Real un1 = n1_rot.dot(n);
		Real un2 = n2_rot.dot(n);
		printf("U * N: %f,%f,%f\n",un0,un1,un2);
		

		/*
		Real l0 = n0_rot.dot(F_i * n0);
		Real l1 = n1_rot.dot(F_i * n1);
		Real l2 = n2_rot.dot(F_i * n2);
		*/
#endif

		matU[pId] = U;
		matV[pId] = V;
		Real l0 = D(0, 0);
		Real l1 = D(1, 1);
		Real l2 = D(2, 2);

		const Real slimit = min(t, strainLimit);
		l0 = clamp(l0, slimit, 1/slimit);
		l1 = clamp(l1, slimit, 1/slimit);
		l2 = clamp(l2, slimit, 1/slimit);
		/*
		if (l0 == slimit)
			printf("l0 got low clamp, %f\n",l0);
		else if(l0 == 1/slimit)
			printf("l0 got high clamp, %f\n",l0);
		if (l1 == slimit)
			printf("l1 got low clamp, %f\n",l1);
		else if (l1 == 1 / slimit)
			printf("l1 got high clamp, %f\n",l1);
		if (l2 == slimit)
			printf("l2 got low clamp, %f\n",l2);
		else if (l2 == 1 / slimit)
			printf("l2 got high clamp, %f\n",l2);
			*/

		D(0, 0) = l0;
		D(1, 1) = l1;
		D(2, 2) = l2;
		
		eigens[pId] = Coord(D(0,0), D(1,1), D(2,2));
		F[pId] = U * D * V.transpose();
		
	}


	template <typename Real, typename Coord, typename Matrix, typename Bond>
	__global__ void HM_JacobiStepNonsymmetric(
		DArray<Coord> source,
		DArray<Matrix> A,
		DArray<Coord> X,
		DArray<Coord> y_pre,
		DArray<Matrix> matU,
		DArray<Matrix> matV,
		DArray<Matrix> matR,
		DArray<Coord> eigen,
		DArray<bool> validOfK,
		DArray<Matrix> F,
		Real k_bend,
		DArrayList<Bond> bonds,
		Real horizon,
		DArray<Real> volume,
		Real dt,
		EnergyType type,
		bool NeighborSearchingAdjacent)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_pre.size()) return;

		Coord x_i = X[pId];
		int size_i = bonds[pId].size();
	
		Real maxDist = Real(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			Bond bond_ij = bonds[pId][ne];
			int j = bond_ij.idx;
			Coord x_j = X[j];
			Real r = (x_i - x_j).norm();

			maxDist = max(maxDist, r);
		}
		maxDist = maxDist < EPSILON ? Real(1) : maxDist;

		Real kappa = 16 / (15 * M_PI * maxDist * maxDist * maxDist * maxDist * maxDist);

		Real lambda_i1 = eigen[pId][0];
		Real lambda_i2 = eigen[pId][1];
		Real lambda_i3 = eigen[pId][2];

		Matrix U_i = matU[pId];
		Matrix V_i = matV[pId];
		Matrix S1_i;
		Matrix S2_i;
		Real vol = volume[pId];
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
		else if (type == Fiber)
		{
			S1_i = ENERGY_FUNC.fiberModel.getStressTensorPositive(lambda_i1, lambda_i2, lambda_i3, V_i);
			S2_i = ENERGY_FUNC.fiberModel.getStressTensorNegative(lambda_i1, lambda_i2, lambda_i3, V_i);
			ENERGY_FUNC.fiberModel.getInfo(lambda_i1, lambda_i2, lambda_i3, V_i);
		}

		Matrix PK1_i = U_i * S1_i * U_i.transpose();
		Matrix PK2_i = U_i * S2_i * V_i.transpose();

		Real Vol_i = volume[pId];

		Matrix F_i = F[pId];

		Coord y_pre_i = y_pre[pId];

		bool K_valid = validOfK[pId];

		Matrix mat_i(0);
		Coord source_i(0);

		for (int ne = 0; ne < size_i; ne++)
		{
			Bond bond_ij = bonds[pId][ne];
			int j = bond_ij.idx;
			Coord y_pre_j = y_pre[j];
			Coord x_j = X[j];
			Real r = (x_j - x_i).norm();

			if (r > EPSILON)
			{
				Real weight_ij = 1.;
			
				Real Vol_j = volume[j];

				Coord y_ij = y_pre_i - y_pre_j;


				Real lambda = y_ij.norm() / r;

				const Real scale = Vol_i * Vol_j * kappa;
				const Real sw_ij = dt * dt * scale * weight_ij;

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
				else if (type == Fiber) {
					S1_ij = ENERGY_FUNC.fiberModel.getStressTensorPositive(lambda, lambda,lambda, V_i);
					S2_ij = ENERGY_FUNC.fiberModel.getStressTensorNegative(lambda, lambda,lambda, V_i);
				}

				Matrix PK1_ij = K_valid_ij ? PK1_i : S1_ij;
				Matrix PK2_ij = K_valid_ij ? PK2_i : S2_ij;

				if (NeighborSearchingAdjacent)
				{
					Real standardVol = 0.0025f * 0.005f;

					PK1_ij *= (x_i - x_j).normSquared() / standardVol;
					PK2_ij *= (x_i - x_j).normSquared() / standardVol;
				}
				
				Coord dir_ij = lambda > EPSILON ? y_ij.normalize() : Coord(1, 0, 0);

				Coord x_ij = K_valid_ij ? x_i - x_j : dir_ij * (x_i - x_j).norm();

				Matrix F_i_1 = F_i.inverse();
				Matrix F_i_T = F_i_1.transpose();
			
				Matrix F_TF_1 = k_bend * F_i_T * F_i_1;

				Coord ds_ij = sw_ij * (PK1_ij + F_TF_1) * y_pre_j + sw_ij * (PK2_ij + k_bend * F_i_1) * x_ij;
				Coord ds_ji = sw_ij * (PK1_ij + F_TF_1) * y_pre_i - sw_ij * (PK2_ij + k_bend * F_i_1) * x_ij;

				Matrix mat_ij = sw_ij * (PK1_ij + F_TF_1);
				

				source_i += ds_ij;

				mat_i += (mat_ij); 
			

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

	

	template<typename Matrix, typename Coord, typename Real>
	__global__ void HM_ComputeNextPosition(
		DArray<Coord> y_next,
		DArray<Coord> y_inter,
		DArray<Real> volume,
		DArray<Coord> source,
		DArray<Matrix> A
		)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		Real mass_i = volume[pId] * 1000.0;
		Matrix mat_i = A[pId] + mass_i * Matrix::identityMatrix();
		Coord src_i = source[pId] + mass_i * y_inter[pId];
		y_next[pId] = mat_i.inverse() * src_i;
	}


	template <typename Real>
	__global__ void HM_InitVolume(
		DArray<Real> volume,
		Real objectVolume,
		bool objectVolumeSet,
		Real particleVolume,
		bool particleVolumeSet
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= volume.size()) return;

		if (objectVolumeSet && volume.size() != 0) volume[pId] = Real(objectVolume / volume.size());
		else if (particleVolumeSet) volume[pId] = particleVolume;
		else volume[pId] = 0.001;
	}


	template <typename Coord, typename Real>
	__global__ void HM_ComputeGradient(
		DArray<Coord> grad,
		DArray<Real> grad_m,
		DArray<Coord> y_pre,
		DArray<Coord> y_next)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= y_next.size()) return;

		grad[pId] = y_next[pId] - y_pre[pId];
		grad_m[pId] = sqrt(grad[pId].dot(grad[pId]));
	}


	template<typename TDataType>
	void CoSemiImplicitHyperelasticitySolver<TDataType>::resizeAllFields()
	{

		uint num = this->inY()->getData().size();

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
		m_volume.resize(num);

		m_energy.resize(num);
		m_alpha.resize(num);
		m_gradient.resize(num);
		mEnergyGradient.resize(num);

		y_pre.resize(num);
		y_current.resize(num);
		y_gradC.resize(num);

		m_source.resize(num);         
		m_A.resize(num);
		m_gradientMagnitude.resize(num);
		mPosBuf.resize(num);

		m_fraction.resize(num);

		m_bFixed.resize(num);
		m_fixedPos.resize(num);
        mPosBuf_March.resize(num);

		initializeVolume();

		cuExecute(m_matU.size(),
			HM_InitRotation,
			m_matU,
			m_matV,
			m_matR);

		m_reduce = Reduction<Real>::Create(num);

		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->inTriangularMesh()->getDataPtr());
		triSet->getPoints().assign(this->inY()->getData());
		triSet->updateAngleWeightedVertexNormal(this->inNorm()->getData());
	}

	template<typename TDataType>
	void CoSemiImplicitHyperelasticitySolver<TDataType>::enforceHyperelasticity()
	{
		resizeAllFields();

		int numOfParticles = this->inY()->getData().size();
		uint pDims = cudaGridSize(numOfParticles, BLOCK_SIZE);

		std::cout << "enforceElasticity Particles: " << numOfParticles << std::endl;

		this->mWeights.reset();

		Log::sendMessage(Log::User, "\n \n \n \n=================== solver start!!!===================");



		/*====================================== Jacobi method ======================================*/
		// initialize y_now, y_next_iter
		y_current.assign(this->inY()->getData());
		this->inMarchPosition()->getData().assign(this->inY()->getData());
		mPosBuf.assign(this->inY()->getData());


		// do Jacobi method Loop
		bool convergeFlag = false; // converge or not
		int iterCount = 0;
		Real omega;
		Real alpha = 1.0f;
		Real max_grad_mag = Real(1000.0);

		if (selfContact) {
			while (iterCount < this->varIterationNumber()->getData() && max_grad_mag >= this->grad_res_eps) {
			
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
					this->inHorizon()->getData(),
					(Real const)0.3,
					this->inRestNorm()->getData(),
					this->inNorm()->getData());
				cuSynchronize();

				HM_JacobiStepNonsymmetric << <pDims, BLOCK_SIZE >> > (
					m_source,
					m_A,
					this->inX()->getData(),
					y_current,
					m_matU,
					m_matV,
					m_matR,
					m_eigenValues,
					m_validOfK,
					m_F,
					this->k_bend,
					this->inBonds()->getData(),
					this->inHorizon()->getData(),
					m_volume,
					this->inTimeStep()->getData(),
					this->inEnergyType()->getData(),
					this->varNeighborSearchingAdjacent()->getData());
				cuSynchronize();

				cuExecute(y_current.size(),
					HM_ComputeNextPosition,
					this->inMarchPosition()->getData(),
					mPosBuf,
					m_volume,
					m_source,
					m_A);

				//sub steping
				cuExecute(m_gradient.size(),
					HM_ComputeGradient,
					m_gradient,
					m_gradientMagnitude,
					y_current,
					this->inMarchPosition()->getData());


				Reduction<Real> reduce;
				max_grad_mag = reduce.maximum(m_gradientMagnitude.begin(), m_gradientMagnitude.size());

				if (this->m_alphaCompute) {
					cuExecute(m_energy.size(),
						HM_Compute1DEnergy,
						m_energy,
						mEnergyGradient,
						this->inX()->getData(),
						y_current,
						m_F,
						m_volume,
						m_validOfK,
						m_eigenValues,
						this->inBonds()->getData(),
						this->inEnergyType()->getData());

					cuExecute(m_alpha.size(),
						HM_ComputeStepLength,
						m_alpha,
						m_gradient,
						mEnergyGradient,
						m_volume,
						m_A,
						m_energy,
						this->inBonds()->getData());

					cuExecute(m_gradient.size(),
						HM_ComputeCurrentPosition,
						m_gradient,
						y_current,
						this->inMarchPosition()->getData(),
						m_alpha);
				}
				else {
					alpha = Real(1.0);
					cuExecute(m_gradient.size(),
						HM_ComputeCurrentPosition,
						m_gradient,
						y_current,
						this->inMarchPosition()->getData(),
						alpha);
				}

				iterCount++;

				y_current.assign(this->inMarchPosition()->getData());
			}

			// do Jacobi method Loop
			mContactRule->initCCDBroadPhase();

			this->inMarchPosition()->getData().assign(mPosBuf);
			convergeFlag = false; // converge or not
			iterCount = 0;
			alpha = 1.0f;
			max_grad_mag = 1e3;
	}
		mPosBuf_March.assign(mPosBuf);

		while (iterCount < this->varIterationNumber()->getData() && max_grad_mag >= this->grad_res_eps) {
			m_source.reset();
			m_A.reset();
			y_current.assign(this->inMarchPosition()->getData());

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
				this->inHorizon()->getData(),
				(Real const)0.3,
				this->inRestNorm()->getData(),
				this->inNorm()->getData());
			cuSynchronize();

			HM_JacobiStepNonsymmetric << <pDims, BLOCK_SIZE >> > (
				m_source,
				m_A,
				this->inX()->getData(),
				y_current,
				m_matU,
				m_matV,
				m_matR,
				m_eigenValues,
				m_validOfK,
				m_F,
				this->k_bend,
				this->inBonds()->getData(),
				this->inHorizon()->getData(),
				m_volume,
				this->inTimeStep()->getData(),
				this->inEnergyType()->getData(),
				this->varNeighborSearchingAdjacent()->getData());
			cuSynchronize();

			cuExecute(this->inMarchPosition()->getData().size(),
				HM_ComputeNextPosition,
				this->inMarchPosition()->getData(),
				mPosBuf_March,
				m_volume,
				m_source,
				m_A);

			//sub steping
			cuExecute(m_gradient.size(),
				HM_ComputeGradient,
				m_gradient,
				m_gradientMagnitude,
				y_current,
				this->inMarchPosition()->getData());

			Reduction<Real> reduce;
			max_grad_mag = reduce.maximum(m_gradientMagnitude.begin(), m_gradientMagnitude.size());

			if (this->m_alphaCompute) {
				cuExecute(m_energy.size(),
					HM_Compute1DEnergy,
					m_energy,
					mEnergyGradient,
					this->inX()->getData(),
					y_current,
					m_F,
					m_volume,
					m_validOfK,
					m_eigenValues,
					this->inBonds()->getData(),
					this->inEnergyType()->getData());

				cuExecute(m_alpha.size(),
					HM_ComputeStepLength,
					m_alpha,
					m_gradient,
					mEnergyGradient,
					m_volume,
					m_A,
					m_energy,
					this->inBonds()->getData());

				cuExecute(m_gradient.size(),
					HM_ComputeCurrentPosition,
					m_gradient,
					y_current,
					this->inMarchPosition()->getData(),
					m_alpha);
			}
			else {
				alpha = Real(1.0);

				cuExecute(m_gradient.size(),
					HM_ComputeCurrentPosition,
					m_gradient,
					y_current,
					this->inMarchPosition()->getData(),
					alpha);
			}
			if (this->selfContact) {
				if (this->acc) {
					mContactRule->update();
				}
				else {
					mContactRule->constrain();
				}
			}

			iterCount++;
		}
		
		/*========================= end of alg, marching time step==============================*/
		printf("========= Enforcement elastic run %d iteration =======\n", iterCount);
	

		cuExecute(this->inY()->getDataPtr()->size(),
			test_HM_UpdatePosition,
			this->inY()->getData(),
			this->inVelocity()->getData(),
			this->inMarchPosition()->getData(),
			mPosBuf,
			this->inAttribute()->getData(),
			this->inTimeStep()->getData());
		

		Log::sendMessage(Log::User, "\n==================== solver end!!!====================\n");
	}

	DEFINE_CLASS(CoSemiImplicitHyperelasticitySolver);
}