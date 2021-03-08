#include <cuda_runtime.h>
#include "ElasticityModule.h"
#include "Framework/Node.h"
#include "Algorithm/MatrixFunc.h"
#include "Utility.h"
#include "Kernel.h"

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
		GArray<Matrix> invK,
		NeighborList<NPair> restShapes)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= invK.size()) return;

		NPair np_i = restShapes.getElement(pId, 0);
		Coord rest_i = np_i.pos;
		int size_i = restShapes.getNeighborSize(pId);

		Real maxDist = Real(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
			int j = np_j.index;
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
			NPair np_j = restShapes.getElement(pId, ne);
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

		

// 		polarDecomposition(mat_i, R, U, D);
// 
// 		Real threshold = 0.0001f*smoothingLength;
// 		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
// 		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
// 		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;
// 
// 		mat_i = R.transpose()*U*D*U.transpose();

// 		printf("Mat: \n %f %f %f \n %f %f %f \n %f %f %f \n	R: \n %f %f %f \n %f %f %f \n %f %f %f \n D: \n %f %f %f \n %f %f %f \n %f %f %f \n U :\n %f %f %f \n %f %f %f \n %f %f %f \n Determinant: %f \n\n",
// 			mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
// 			mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
// 			mat_i(2, 0), mat_i(2, 1), mat_i(2, 2),
// 			R(0, 0), R(0, 1), R(0, 2),
// 			R(1, 0), R(1, 1), R(1, 2),
// 			R(2, 0), R(2, 1), R(2, 2),
// 			D(0, 0), D(0, 1), D(0, 2),
// 			D(1, 0), D(1, 1), D(1, 2),
// 			D(2, 0), D(2, 1), D(2, 2),
// 			U(0, 0), U(0, 1), U(0, 2),
// 			U(1, 0), U(1, 1), U(1, 2),
// 			U(2, 0), U(2, 1), U(2, 2),
// 			R.determinant());
// 		printf("Mat: \n %f %f %f \n %f %f %f \n %f %f %f \n	U :\n %f %f %f \n %f %f %f \n %f %f %f \n Determinant: %f \n\n",
// 			mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
// 			mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
// 			mat_i(2, 0), mat_i(2, 1), mat_i(2, 2),
// 			U(0, 0), U(0, 1), U(0, 2),
// 			U(1, 0), U(1, 1), U(1, 2),
// 			U(2, 0), U(2, 1), U(2, 2),
// 			R.determinant());

		invK[pId] = mat_i;
	}

	__device__ float EM_GetStiffness(int r)
	{
		return 10.0f;
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void EM_EnforceElasticity(
		GArray<Coord> delta_position,
		GArray<Real> weights,
		GArray<Real> bulkCoefs,
		GArray<Matrix> invK,
		GArray<Coord> position,
		NeighborList<NPair> restShapes,
		Real mu,
		Real lambda)
	{

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		NPair np_i = restShapes.getElement(pId, 0);
		Coord rest_i = np_i.pos;
		int size_i = restShapes.getNeighborSize(pId);

		Coord cur_pos_i = position[pId];

		Coord accPos = Coord(0);
		Real accA = Real(0);
		Real bulk_i = bulkCoefs[pId];

		Real maxDist = Real(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
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
			NPair np_j = restShapes.getElement(pId, ne);
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


		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShapes.getElement(pId, ne);
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
		GArray<Coord> position,
		GArray<Coord> old_position,
		GArray<Coord> delta_position,
		GArray<Real> delta_weights)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		position[pId] = (old_position[pId] + delta_position[pId]) / (1.0+delta_weights[pId]);
	}


	template <typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		GArray<Coord> velArr,
		GArray<Coord> prePos,
		GArray<Coord> curPos,
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
//		this->attachField(&m_horizon, "horizon", "Supporting radius!", false);
//		this->attachField(&m_distance, "distance", "The sampling distance!", false);
		this->attachField(&m_mu, "mu", "Material stiffness!", false);
		this->attachField(&m_lambda, "lambda", "Material stiffness!", false);
		this->attachField(&m_iterNum, "Iterations", "Iteration Number", false);

//		this->attachField(&m_position, "position", "Storing the particle positions!", false);
//		this->attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
//		this->attachField(&m_neighborhood, "neighborhood", "Storing neighboring particles' ids!", false);

//		this->attachField(&testing, "testing", "For testing", false);
//		this->attachField(&TetOut, "TetOut", "For testing", false);

		this->inHorizon()->setValue(0.0125);
 		m_mu.setValue(0.05);
 		m_lambda.setValue(0.1);
		m_iterNum.setValue(10);

		this->inNeighborhood()->tagOptional(true);
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
			this->inPosition()->getValue(),
			this->inRestShape()->getValue(),
			m_mu.getValue(),
			m_lambda.getValue());
		cuSynchronize();

		K_UpdatePosition << <pDims, BLOCK_SIZE >> > (
			this->inPosition()->getValue(),
			m_position_old,
			m_displacement,
			m_weights);
		cuSynchronize();
	}

	template<typename Real>
	__global__ void EM_InitBulkStiffness(GArray<Real> stiffness)
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
		int num = this->inRestShape()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		EM_PrecomputeShape <Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (
			m_invK,
			this->inRestShape()->getValue());
		cuSynchronize();
	}


	template<typename TDataType>
	void ElasticityModule<TDataType>::solveElasticity()
	{
		//Save new positions
		Function1Pt::copy(m_position_old, this->inPosition()->getValue());

		this->computeInverseK();

		int itor = 0;
		while (itor < m_iterNum.getValue())
		{
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
			this->inVelocity()->getValue(),
			m_position_old,
			this->inPosition()->getValue(),
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
		NeighborList<NPair> shape,
		NeighborList<int> nbr,
		GArray<Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		NPair np;
		int nbSize = nbr.getNeighborSize(pId);
		
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbr.getElement(pId, ne);
			np.index = j;
			np.pos = pos[j];
			np.weight = 1;
 			if (pId != j)
 			{
// 				if (pId == 4 && j == 5)
// 				{
// 					np.pos += Coord(0.0001, 0, 0);
// 				}
// 
// 				if (pId == 5 && j == 4)
// 				{
// 					np.pos += Coord(-0.0001, 0, 0);
// 				}

 				shape.setElement(pId, ne, np);
			}
			else
			{
				if (ne == 0)
				{
					shape.setElement(pId, ne, np);
				}
				else
				{
					auto ele = shape.getElement(pId, 0);
					shape.setElement(pId, 0, np);
					shape.setElement(pId, ne, ele);
				}
			}
		}
	}

	template<typename TDataType>
	void ElasticityModule<TDataType>::resetRestShape()
	{
		this->inRestShape()->setElementCount(this->inNeighborhood()->getValue().size());
		this->inRestShape()->getValue().getIndex().resize(this->inNeighborhood()->getValue().getIndex().size());

		if (this->inNeighborhood()->getValue().isLimited())
		{
			this->inRestShape()->getValue().setNeighborLimit(this->inNeighborhood()->getValue().getNeighborLimit());
		}
		else
		{
			this->inRestShape()->getValue().getElements().resize(this->inNeighborhood()->getValue().getElements().size());
		}

		Function1Pt::copy(this->inRestShape()->getValue().getIndex(), this->inNeighborhood()->getValue().getIndex());

		uint pDims = cudaGridSize(this->inPosition()->getValue().size(), BLOCK_SIZE);

		K_UpdateRestShape<< <pDims, BLOCK_SIZE >> > (this->inRestShape()->getValue(), this->inNeighborhood()->getValue(), this->inPosition()->getValue());
		cuSynchronize();
	}

	template<typename TDataType>
	bool ElasticityModule<TDataType>::initializeImpl()
	{
		if (this->inHorizon()->isEmpty() || this->inPosition()->isEmpty() || this->inVelocity()->isEmpty() || this->inNeighborhood()->isEmpty())
		{
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

		Function1Pt::copy(m_position_old, this->inPosition()->getValue());

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

		Function1Pt::copy(m_position_old, this->inPosition()->getValue());
	}

}