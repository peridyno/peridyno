#include <cuda_runtime.h>
#include "ElasticityModule.h"
#include "Framework/Node.h"
#include "Matrix/MatrixFunc.h"
#include "ParticleSystem/Kernel.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(ElasticityModule, TDataType)

	template<typename Real>
	__device__ Real D_Weight(Real r, Real h)
	{
		CorrectedKernel<Real> kernSmooth;
		return kernSmooth.WeightRR(r, 2*h);
// 		h = h < EPSILON ? Real(1) : h;
// 		return 1 / (h*h*h);
	}


	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void EM_PrecomputeShape(
		DArray<Matrix> invK,
		DArrayList<NPair> restShapes)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= invK.size()) return;

		List<NPair>& restShape_i = restShapes[pId];
		NPair np_i = restShape_i[0];
		Coord rest_i = np_i.pos;
		int size_i = restShape_i.size();
		Real maxDist = Real(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape_i[ne];
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_i - rest_pos_j).norm();

			maxDist = max(maxDist, r);
		}
		maxDist = maxDist < EPSILON ? Real(1) : maxDist;
		Real smoothingLength = maxDist;

		Real total_weight = 0.0f;
		Matrix mat_i = Matrix(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape_i[ne];
			Coord rest_j = np_j.pos;
			Real r = (rest_i - rest_j).norm();

			if (r > EPSILON)
			{
				Real weight = D_Weight(r, smoothingLength);
				Coord q = (rest_j - rest_i) / smoothingLength*sqrt(weight);

				mat_i(0, 0) += q[0] * q[0]; mat_i(0, 1) += q[0] * q[1]; mat_i(0, 2) += q[0] * q[2];
				mat_i(1, 0) += q[1] * q[0]; mat_i(1, 1) += q[1] * q[1]; mat_i(1, 2) += q[1] * q[2];
				mat_i(2, 0) += q[2] * q[0]; mat_i(2, 1) += q[2] * q[1]; mat_i(2, 2) += q[2] * q[2];

				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			mat_i *= (1.0f / total_weight);
		}

		Matrix R(0), U(0), D(0), V(0);

// 		if (pId == 0)
// 		{
// 			printf("EM_PrecomputeShape**************************************");
// 
// 			printf("K: \n %f %f %f \n %f %f %f \n %f %f %f \n\n\n",
// 				mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
// 				mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
// 				mat_i(2, 0), mat_i(2, 1), mat_i(2, 2));
// 		}

		polarDecomposition(mat_i, R, U, D, V);

		if (mat_i.determinant() < EPSILON*smoothingLength)
		{
			Real threshold = 0.0001f*smoothingLength;
			D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
			D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
			D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;

			mat_i = V * D*U.transpose();
		}
		else
			mat_i = mat_i.inverse();

		invK[pId] = mat_i;
	}

	__device__ float EM_GetStiffness(int r)
	{
		return 10.0f;
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void EM_EnforceElasticity(
		DArray<Coord> delta_position,
		DArray<Real> weights,
		DArray<Real> bulkCoefs,
		DArray<Matrix> invK,
		DArray<Coord> position,
		DArrayList<NPair> restShapes,
		Real mu,
		Real lambda)
	{

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		List<NPair>& restShape_i = restShapes[pId];
		NPair np_i = restShape_i[0];
		Coord rest_i = np_i.pos;
		int size_i = restShape_i.size();

		Coord cur_pos_i = position[pId];

		Coord accPos = Coord(0);
		Real accA = Real(0);
		Real bulk_i = bulkCoefs[pId];

		Real maxDist = Real(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape_i[ne];
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_i - rest_pos_j).norm();

			maxDist = max(maxDist, r);
		}
		maxDist = maxDist < EPSILON ? Real(1) : maxDist;
		Real horizon = maxDist;


		Real total_weight = 0.0f;
		Matrix deform_i = Matrix(0.0f);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape_i[ne];
			Coord rest_j = np_j.pos;
			int j = np_j.index;

			Real r = (rest_j - rest_i).norm();

			if (r > EPSILON)
			{
				Real weight = D_Weight(r, horizon);

				Coord p = (position[j] - position[pId]) / horizon;
				Coord q = (rest_j - rest_i) / horizon*weight;

				deform_i(0, 0) += p[0] * q[0]; deform_i(0, 1) += p[0] * q[1]; deform_i(0, 2) += p[0] * q[2];
				deform_i(1, 0) += p[1] * q[0]; deform_i(1, 1) += p[1] * q[1]; deform_i(1, 2) += p[1] * q[2];
				deform_i(2, 0) += p[2] * q[0]; deform_i(2, 1) += p[2] * q[1]; deform_i(2, 2) += p[2] * q[2];
				total_weight += weight;
			}
		}


		if (total_weight > EPSILON)
		{
			deform_i *= (1.0f / total_weight);
			deform_i = deform_i * invK[pId];
		}
		else
		{
			total_weight = 1.0f;
		}

		//Check whether the reference shape is inverted, if yes, simply set K^{-1} to be an identity matrix
		//Note other solutions are possible.
		if ((deform_i.determinant()) < -0.001f)
		{
			deform_i = Matrix::identityMatrix(); 
		}


// 		//if (pId == 0)
// 		{
// 			Matrix mat_i = invK[pId];
// 			printf("Mat %d: \n %f %f %f \n %f %f %f \n %f %f %f \n", 
// 				pId,
// 				mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
// 				mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
// 				mat_i(2, 0), mat_i(2, 1), mat_i(2, 2));
// 		}

		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape_i[ne];
			Coord rest_j = np_j.pos;
			int j = np_j.index;

			Coord cur_pos_j = position[j];
			Real r = (rest_j - rest_i).norm();

			if (r > 0.01f*horizon)
			{
				Real weight = D_Weight(r, horizon);

				Coord rest_dir_ij = deform_i*(rest_i - rest_j);
				Coord cur_dir_ij = cur_pos_i - cur_pos_j;

				cur_dir_ij = cur_dir_ij.norm() > EPSILON ? cur_dir_ij.normalize() : Coord(0);
				rest_dir_ij = rest_dir_ij.norm() > EPSILON ? rest_dir_ij.normalize() : Coord(0, 0, 0);

				Real mu_ij = mu*bulk_i* D_Weight(r, horizon);
				Coord mu_pos_ij = position[j] + r*rest_dir_ij;
				Coord mu_pos_ji = position[pId] - r*rest_dir_ij;

				Real lambda_ij = lambda*bulk_i*D_Weight(r, horizon);
				Coord lambda_pos_ij = position[j] + r*cur_dir_ij;
				Coord lambda_pos_ji = position[pId] - r*cur_dir_ij;

				Coord delta_pos_ij = mu_ij*mu_pos_ij + lambda_ij*lambda_pos_ij;
				Real delta_weight_ij = mu_ij + lambda_ij;

				Coord delta_pos_ji = mu_ij*mu_pos_ji + lambda_ij*lambda_pos_ji;

				accA += delta_weight_ij;
				accPos += delta_pos_ij;


				atomicAdd(&weights[j], delta_weight_ij);
				atomicAdd(&delta_position[j][0], delta_pos_ji[0]);
				atomicAdd(&delta_position[j][1], delta_pos_ji[1]);
				atomicAdd(&delta_position[j][2], delta_pos_ji[2]);
			}
		}

		atomicAdd(&weights[pId], accA);
		atomicAdd(&delta_position[pId][0], accPos[0]);
		atomicAdd(&delta_position[pId][1], accPos[1]);
		atomicAdd(&delta_position[pId][2], accPos[2]);
	}

	template <typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DArray<Coord> position,
		DArray<Coord> old_position,
		DArray<Coord> delta_position,
		DArray<Real> delta_weights)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		position[pId] = (old_position[pId] + delta_position[pId]) / (1.0+delta_weights[pId]);

		if (pId == 0)
		{
			Coord vec_i = position[pId];
			printf("Vec: \n %f %f %f \n",
				vec_i[0], vec_i[1], vec_i[2]);
		}
	}


	template <typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DArray<Coord> velArr,
		DArray<Coord> prePos,
		DArray<Coord> curPos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;

		velArr[pId] += (curPos[pId] - prePos[pId]) / dt;
	}

	template<typename TDataType>
	ElasticityModule<TDataType>::ElasticityModule()
		: ConstraintModule()
	{
		this->inHorizon()->setValue(0.0125);
 		m_mu.setValue(0.02);
 		m_lambda.setValue(0.01);
		m_iterNum.setValue(10);

		this->inNeighborIds()->tagOptional(true);
	}


	template<typename TDataType>
	ElasticityModule<TDataType>::~ElasticityModule()
	{
		m_weights.clear();
		m_displacement.clear();
		m_invK.clear();
		m_F.clear();
		m_position_old.clear();
	}

	template<typename TDataType>
	void ElasticityModule<TDataType>::enforceElasticity()
	{
		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		m_displacement.reset();
		m_weights.reset();

		EM_EnforceElasticity << <pDims, BLOCK_SIZE >> > (
			m_displacement,
			m_weights,
			m_bulkCoefs,
			m_invK,
			this->inPosition()->getData(),
			this->inRestShape()->getData(),
			m_mu.getData(),
			m_lambda.getData());
		cuSynchronize();

		K_UpdatePosition << <pDims, BLOCK_SIZE >> > (
			this->inPosition()->getData(),
			m_position_old,
			m_displacement,
			m_weights);
		cuSynchronize();
	}

	template<typename Real>
	__global__ void EM_InitBulkStiffness(DArray<Real> stiffness)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= stiffness.size()) return;

		stiffness[pId] = Real(1);
	}

	template<typename TDataType>
	void ElasticityModule<TDataType>::computeMaterialStiffness()
	{
		int num = this->inPosition()->getElementCount();

		uint pDims = cudaGridSize(num, BLOCK_SIZE);
		EM_InitBulkStiffness << <pDims, BLOCK_SIZE >> > (m_bulkCoefs);
	}


	template<typename TDataType>
	void ElasticityModule<TDataType>::computeInverseK()
	{
		auto& restShapes = this->inRestShape()->getData();
		uint pDims = cudaGridSize(restShapes.size(), BLOCK_SIZE);

		EM_PrecomputeShape <Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (
			m_invK,
			restShapes);
		cuSynchronize();
	}


	template<typename TDataType>
	void ElasticityModule<TDataType>::solveElasticity()
	{
		//Save new positions
		m_position_old.assign(this->inPosition()->getData());

		this->computeInverseK();

		int itor = 0;
		while (itor < m_iterNum.getData()) {
			this->enforceElasticity();
			itor++;
		}

		this->updateVelocity();
	}

	template<typename TDataType>
	void ElasticityModule<TDataType>::updateVelocity()
	{
		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		Real dt = this->getParent()->getDt();

		K_UpdateVelocity << <pDims, BLOCK_SIZE >> > (
			this->inVelocity()->getData(),
			m_position_old,
			this->inPosition()->getData(),
			dt);
		cuSynchronize();
	}


	template<typename TDataType>
	bool ElasticityModule<TDataType>::constrain()
	{
		this->solveElasticity();

		return true;
	}


	template <typename Coord, typename NPair>
	__global__ void K_UpdateRestShape(
		DArrayList<NPair> shape,
		DArrayList<int> nbr,
		DArray<Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		NPair np;

		List<NPair>& rest_shape_i = shape[pId];
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
				NPair np_0 = rest_shape_i[0];
				rest_shape_i[0] = np;
				rest_shape_i[ne] = np_0;
			}
		}

		if (pId == 0)
		{
			for (int ne = 0; ne < nbSize; ne++)
			{
				NPair np_j = rest_shape_i[ne];

				printf("%d: %d; %f %f %f \n", pId, np_j.index, np_j.pos[0], np_j.pos[1], np_j.pos[2]);
			}
		}
		

// 		for (int ne = 0; ne < nbSize; ne++)
// 		{
// 			int j = list_id_i[ne];
// 
// 			printf("%d: %d; %f %f %f \n", pId, j, pos[j][0], pos[j][1], pos[j][2]);
// 		}
	}

	template<typename TDataType>
	void ElasticityModule<TDataType>::resetRestShape()
	{
		auto& nbrIds = this->inNeighborIds()->getData();

		if (this->inRestShape()->isEmpty()) {
			this->inRestShape()->allocate();
		}

		this->inRestShape()->getDataPtr()->resize(nbrIds);

		cuExecute(nbrIds.size(),
			K_UpdateRestShape,
			this->inRestShape()->getData(),
			this->inNeighborIds()->getData(),
			this->inPosition()->getData());
	}

	template<typename TDataType>
	bool ElasticityModule<TDataType>::initializeImpl()
	{
		if (this->inHorizon()->isEmpty() || this->inPosition()->isEmpty() || this->inVelocity()->isEmpty() || this->inNeighborIds()->isEmpty())
		{
			if (this->inHorizon()->isEmpty())
			{
				return false;
			}

			if (this->inPosition()->isEmpty())
			{
				return false;
			}

			if (this->inVelocity()->isEmpty())
			{
				return false;
			}

			if (this->inNeighborIds()->isEmpty())
			{
				return false;
			}

			std::cout << "Exception: " << std::string("ElasticityModule's fields are not fully initialized!") << "\n";
			return false;
		}

		int num = this->inPosition()->getElementCount();
		
		m_invK.resize(num);
		m_weights.resize(num);
		m_displacement.resize(num);

		m_F.resize(num);
		
		m_position_old.resize(num);
		m_bulkCoefs.resize(num);

		if (this->inRestShape()->isEmpty())
		{
			resetRestShape();
		}
		
		this->computeMaterialStiffness();

		m_position_old.assign(this->inPosition()->getData());

		return true;
	}


	template<typename TDataType>
	void ElasticityModule<TDataType>::begin()
	{
		int num = this->inPosition()->getElementCount();

		if (num == m_invK.size())
			return;

		m_invK.resize(num);
		m_weights.resize(num);
		m_displacement.resize(num);

		m_F.resize(num);

		m_position_old.resize(num);
		m_bulkCoefs.resize(num);

		this->computeMaterialStiffness();

		m_position_old.assign(this->inPosition()->getData());
	}

	DEFINE_CLASS(ElasticityModule);
}