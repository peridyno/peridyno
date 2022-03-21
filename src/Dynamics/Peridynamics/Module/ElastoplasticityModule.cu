#include <cuda_runtime.h>
#include "ElastoplasticityModule.h"
#include "Node.h"
#include "Matrix/MatrixFunc.h"
#include "ParticleSystem/Kernel.h"
#include <thrust/scan.h>
#include <thrust/reduce.h>
//#include "svd3_cuda2.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ElastoplasticityModule, TDataType)

	template<typename TDataType>
	ElastoplasticityModule<TDataType>::ElastoplasticityModule()
		: ElasticityModule<TDataType>()
	{
		this->attachField(&m_c, "c", "cohesion!", false);
		this->attachField(&m_phi, "phi", "friction angle!", false);

		m_c.setValue(0.001);
		m_phi.setValue(60.0 / 180.0);

		m_reconstuct_all_neighborhood.setValue(false);
		m_incompressible.setValue(true);
	}

	__device__ Real Hardening(Real rho)
	{
		Real hardening = 1.0f;

		return 1.0f;

		if (rho >= 1000)
		{
			Real ratio = rho / 1000;
			return pow((float)M_E, hardening*(ratio - 1.0f));
		}
		else
			return 1.0f;
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void PM_ComputeInvariants(
		DArray<bool> bYield,
		DArray<Real> yield_I1,
		DArray<Real> yield_J2,
		DArray<Real> arrI1,
		DArray<Coord> position,
		DArray<Real> density,
		DArray<Real> bulk_stiffiness,
		DArrayList<NPair> restShape,
		Real horizon,
		Real A,
		Real B,
		Real mu,
		Real lambda)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= position.size()) return;

		CorrectedKernel<Real> kernSmooth;

		Real weaking = 1.0f;// Softening(rhoArr[i]);

		Real s_A = weaking*A;

		List<NPair>& rest_shape_i = restShape[i];
		Coord rest_pos_i = rest_shape_i[0].pos;
		Coord cur_pos_i = position[i];

		Real I1_i = 0.0f;
		Real J2_i = 0.0f;
		//compute the first and second invariants of the deformation state, i.e., I1 and J2
		int size_i = rest_shape_i.size();
		Real total_weight = Real(0);
		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = rest_shape_i[ne];
			Coord rest_pos_j = np_j.pos;
			int j = np_j.index;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > 0.01*horizon)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				Coord p = (position[j] - cur_pos_i);
				Real ratio_ij = p.norm() / r;

				I1_i += weight*ratio_ij;

				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			I1_i /= total_weight;
		}
		else
		{
			I1_i = 1.0f;
		}

		for (int ne = 1; ne < size_i; ne++)
		{
			NPair np_j = rest_shape_i[ne];
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > 0.01*horizon)
			{
				Real weight = kernSmooth.Weight(r, horizon);
				Vec3f p = (position[j] - cur_pos_i);
				Real ratio_ij = p.norm() / r;
				J2_i = (ratio_ij - I1_i)*(ratio_ij - I1_i)*weight;
			}
		}
		if (total_weight > EPSILON)
		{
			J2_i /= total_weight;
			J2_i = sqrt(J2_i);
		}
		else
		{
			J2_i = 0.0f;
		}

		Real D1 = 1 - I1_i;		//positive for compression and negative for stretching

		Real yield_I1_i = 0.0f;
		Real yield_J2_i = 0.0f;

		Real s_J2 = J2_i*mu*bulk_stiffiness[i];
		Real s_D1 = D1*lambda*bulk_stiffiness[i];

		//Drucker-Prager yield criterion
		if (s_J2 <= s_A + B*s_D1)
		{
			//bulk_stiffiness[i] = 10.0f;
			//invDeform[i] = Matrix::identityMatrix();
			yield_I1[i] = Real(0);
			yield_J2[i] = Real(0);

			bYield[i] = false;
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

			yield_I1[i] = yield_I1_i;
			yield_J2[i] = yield_J2_i;
		}
		arrI1[i] = I1_i;
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void PM_ApplyYielding(
		DArray<Real> yield_I1,
		DArray<Real> yield_J2,
		DArray<Real> arrI1,
		DArray<Coord> position,
		DArrayList<NPair> restShape)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= position.size()) return;

		List<NPair>& rest_shape_i = restShape[i];
		Coord rest_pos_i = rest_shape_i[0].pos;
		Coord pos_i = position[i];

		Real yield_I1_i = yield_I1[i];
		Real yield_J2_i = yield_J2[i];
		Real I1_i = arrI1[i];

		//add permanent deformation
		int size_i = rest_shape_i.size();
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = rest_shape_i[ne];
			Coord rest_pos_j = np_j.pos;
			int j = np_j.index;

			Real yield_I1_j = yield_I1[j];
			Real yield_J2_j = yield_J2[j];
			Real I1_j = arrI1[j];

			Real r = (rest_pos_i - rest_pos_j).norm();

			Coord p = (position[j] - pos_i);
			Coord q = (rest_pos_j - rest_pos_i);

			//Coord new_q = q*I1_i;
			Coord new_q = q*(I1_i + I1_j) / 2;
			Coord D_iso = new_q - q;

			Coord dir_q = q;
			dir_q = dir_q.norm() > EPSILON ? dir_q.normalize() : Coord(0);

			Coord D_dev = p.norm()*dir_q - new_q;
			//Coord D_dev = p - new_q;

			NPair new_np_j;

			//Coord new_rest_pos_j = rest_pos_j + yield_I1_i * D_iso + yield_J2_i * D_dev;
			Coord new_rest_pos_j = rest_pos_j + (yield_I1_i + yield_I1_j) / 2 * D_iso + (yield_J2_i + yield_J2_j) / 2 * D_dev;

			new_np_j.pos = new_rest_pos_j;
			new_np_j.index = j;
			rest_shape_i[ne] = new_np_j;
		}

	}


	//	int iter = 0;
	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::constrain()
	{
		this->solveElasticity();
		this->applyPlasticity();
	}


	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::solveElasticity()
	{
		this->mPosBuf.assign(this->inPosition()->getData());

		this->computeInverseK();

		m_pbdModule->varIterationNumber()->setValue(1);

		int iter = 0;
		int total = this->varIterationNumber()->getData();
		while (iter < total)
		{
			this->enforceElasticity();
			if (m_incompressible.getData() == true)
			{
				m_pbdModule->constrain();
			}
			
			iter++;
		}

		this->updateVelocity();
	}

	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::applyPlasticity()
	{
		this->rotateRestShape();

		this->computeMaterialStiffness();
		this->applyYielding();

		this->reconstructRestShape();
	}


	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::applyYielding()
	{
		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		Real A = computeA();
		Real B = computeB();

		PM_ComputeInvariants<Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (
			m_bYield,
			m_yiled_I1,
			m_yield_J2,
			m_I1,
			this->inPosition()->getData(),
			m_pbdModule->outDensity()->getData(),
			this->mBulkStiffness,
			this->inRestShape()->getData(),
			this->inHorizon()->getData(),
			A,
			B,
			this->varMu()->getData(),
			this->varLambda()->getData());
		cuSynchronize();
		// 
		PM_ApplyYielding<Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (
			m_yiled_I1,
			m_yield_J2,
			m_I1,
			this->inPosition()->getData(),
			this->inRestShape()->getData());
		cuSynchronize();
	}


	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void PM_ReconstructRestShape(
		DArrayList<NPair> new_rest_shape,
		DArray<bool> bYield,
		DArray<Coord> position,
		DArray<Real> I1,
		DArray<Real> I1_yield,
		DArray<Real> J2_yield,
		DArray<Matrix> invF,
		DArrayList<int> neighborhood,
		DArrayList<NPair> restShape,
		Real horizon)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= new_rest_shape.size()) return;

		List<int>& list_i = neighborhood[i];
		List<NPair>& list_np_i = restShape[i];
		List<NPair>& new_list_np_i = new_rest_shape[i];

		// update neighbors
		if (!bYield[i])
		{
			Coord rest_pos_i = list_np_i[0].pos;

			int new_size = list_np_i.size();
			for (int ne = 0; ne < new_size; ne++)
			{
				NPair pair = list_np_i[ne];
				new_list_np_i.insert(pair);
			}
		}
		else
		{
			int nbSize = list_i.size();
			Coord pos_i = position[i];

			Matrix invF_i = invF[i];

			NPair np;
			for (int ne = 0; ne < nbSize; ne++)
			{
				int j = list_i[ne];
				Matrix invF_j = invF[j];

				np.index = j;
				np.pos = pos_i + 0.5*(invF_i + invF_j)*(position[j] - pos_i);

				new_list_np_i.insert(np);

				if (i == j)
				{
					NPair np_0 = new_list_np_i[0];
					new_list_np_i[0] = np;
					new_list_np_i[ne] = np_0;
				}
			}
		}

		bYield[i] = false;
	}

	template <typename NPair>
	__global__ void PM_ReconfigureRestShape(
		DArray<int> nbSize,
		DArray<bool> bYield,
		DArrayList<int> neighborhood,
		DArrayList<NPair> restShape)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= nbSize.size()) return;

		if (bYield[i]) {
			nbSize[i] = neighborhood[i].size();
		}
		else {
			nbSize[i] = restShape[i].size();
		}
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void PM_ComputeInverseDeformation(
		DArray<Matrix> invF,
		DArray<Coord> position,
		DArrayList<NPair> restShape,
		Real horizon)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= invF.size()) return;

		CorrectedKernel<Real> kernSmooth;

		//reconstruct the rest shape as the yielding condition is violated.
		Real total_weight = 0.0f;
		Matrix curM(0);
		Matrix refM(0);

		List<NPair>& list_np_i = restShape[i];
		Coord rest_pos_i = list_np_i[0].pos;
		int size_i = list_np_i.size();
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = list_np_i[ne];
			int j = np_j.index;
			Coord rest_j = np_j.pos;
			Real r = (rest_j - rest_pos_i).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, horizon);

				Coord p = (position[j] - position[i]) / horizon;
				Coord q = (rest_j - rest_pos_i) / horizon;

				curM(0, 0) += p[0] * p[0] * weight; curM(0, 1) += p[0] * p[1] * weight; curM(0, 2) += p[0] * p[2] * weight;
				curM(1, 0) += p[1] * p[0] * weight; curM(1, 1) += p[1] * p[1] * weight; curM(1, 2) += p[1] * p[2] * weight;
				curM(2, 0) += p[2] * p[0] * weight; curM(2, 1) += p[2] * p[1] * weight; curM(2, 2) += p[2] * p[2] * weight;

				refM(0, 0) += q[0] * p[0] * weight; refM(0, 1) += q[0] * p[1] * weight; refM(0, 2) += q[0] * p[2] * weight;
				refM(1, 0) += q[1] * p[0] * weight; refM(1, 1) += q[1] * p[1] * weight; refM(1, 2) += q[1] * p[2] * weight;
				refM(2, 0) += q[2] * p[0] * weight; refM(2, 1) += q[2] * p[1] * weight; refM(2, 2) += q[2] * p[2] * weight;

				total_weight += weight;
			}
		}


		if (total_weight < EPSILON)
		{
			total_weight = Real(1);
		}
		refM *= (1.0f / total_weight);
		curM *= (1.0f / total_weight);

		Real threshold = Real(0.00001);
		Matrix curR, curU, curD, curV;

		polarDecomposition(curM, curR, curU, curD, curV);

		curD(0, 0) = curD(0, 0) > threshold ? 1.0 / curD(0, 0) : 1.0 / threshold;
		curD(1, 1) = curD(1, 1) > threshold ? 1.0 / curD(1, 1) : 1.0 / threshold;
		curD(2, 2) = curD(2, 2) > threshold ? 1.0 / curD(2, 2) : 1.0 / threshold;
		refM *= curV*curD*curU.transpose();

		// 		if (abs(refM.determinant() - 1) > 0.05f)
		// 		{
		// 			refM = Matrix::identityMatrix();
		// 		}

		if (refM.determinant() < EPSILON)
		{
			refM = Matrix::identityMatrix();
		}

		invF[i] = refM;
	}

	__global__ void PM_EnableAllReconstruction(
		DArray<bool> bYield)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= bYield.size()) return;

		bYield[i] = true;
	}

	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::reconstructRestShape()
	{
		//constructRestShape(m_neighborhood.getData(), m_position.getData());

		auto& pts = this->inPosition()->getData();
		
		if (m_reconstuct_all_neighborhood.getData())
		{
			cuExecute(m_bYield.size(),
				PM_EnableAllReconstruction,
				m_bYield);
		}


		DArray<int> index(this->inPosition()->getDataPtr()->size());

		cuExecute(index.size(),
			PM_ReconfigureRestShape,
			index,
			m_bYield,
			this->inNeighborIds()->getData(),
			this->inRestShape()->getData());


		DArrayList<NPair> newNeighborList;
		newNeighborList.resize(index);

		cuExecute(m_invF.size(),
			PM_ComputeInverseDeformation,
			m_invF,
			this->inPosition()->getData(),
			this->inRestShape()->getData(),
			this->inHorizon()->getData());

		cuExecute(newNeighborList.size(),
			PM_ReconstructRestShape,
			newNeighborList,
			m_bYield,
			this->inPosition()->getData(),
			m_I1,
			m_yiled_I1,
			m_yield_J2,
			m_invF,
			this->inNeighborIds()->getData(),
			this->inRestShape()->getData(),
			this->inHorizon()->getData());

		this->inRestShape()->getDataPtr()->assign(newNeighborList);

		newNeighborList.clear();
		index.clear();
		cuSynchronize();
	}


	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::enableFullyReconstruction()
	{
		m_reconstuct_all_neighborhood.setValue(true);
	}



	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::disableFullyReconstruction()
	{
		m_reconstuct_all_neighborhood.setValue(false);
	}


	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::enableIncompressibility()
	{
		m_incompressible.setValue(true);
	}

	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::disableIncompressibility()
	{
		m_incompressible.setValue(false);
	}


	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void EM_RotateRestShape(
		DArray<Coord> position,
		DArray<bool> bYield,
		DArrayList<NPair> restShapes,
		Real smoothingLength)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		SmoothKernel<Real> kernSmooth;

		List<NPair>& rest_shape_i = restShapes[pId];
		Coord rest_pos_i = rest_shape_i[0].pos;
		int size_i = rest_shape_i.size();

		//			cout << i << " " << rids[shape_i.ids[shape_i.idx]] << endl;
		Real total_weight = 0.0f;
		Matrix mat_i(0);
		Matrix invK_i(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = rest_shape_i[ne];
			int j = np_j.index;
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_pos_i - rest_pos_j).norm();

			if (r > EPSILON)
			{
				Real weight = kernSmooth.Weight(r, smoothingLength);

				Coord p = (position[j] - position[pId]) / smoothingLength;
				//Vec3f q = (shape_i.pos[ne] - rest_i)*(1.0f/r)*weight;
				Coord q = (rest_pos_j - rest_pos_i) / smoothingLength;

				mat_i(0, 0) += p[0] * q[0] * weight; mat_i(0, 1) += p[0] * q[1] * weight; mat_i(0, 2) += p[0] * q[2] * weight;
				mat_i(1, 0) += p[1] * q[0] * weight; mat_i(1, 1) += p[1] * q[1] * weight; mat_i(1, 2) += p[1] * q[2] * weight;
				mat_i(2, 0) += p[2] * q[0] * weight; mat_i(2, 1) += p[2] * q[1] * weight; mat_i(2, 2) += p[2] * q[2] * weight;

				invK_i(0, 0) += q[0] * q[0] * weight; invK_i(0, 1) += q[0] * q[1] * weight; invK_i(0, 2) += q[0] * q[2] * weight;
				invK_i(1, 0) += q[1] * q[0] * weight; invK_i(1, 1) += q[1] * q[1] * weight; invK_i(1, 2) += q[1] * q[2] * weight;
				invK_i(2, 0) += q[2] * q[0] * weight; invK_i(2, 1) += q[2] * q[1] * weight; invK_i(2, 2) += q[2] * q[2] * weight;

				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			mat_i *= (1.0f / total_weight);
			invK_i *= (1.0f / total_weight);
		}

		Matrix R, U, D, V;
		polarDecomposition(invK_i, R, U, D, V);

		Real threshold = 0.0001f*smoothingLength;
		D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
		D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
		D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;

		invK_i = V*D*U.transpose();

		mat_i *= invK_i;

		polarDecomposition(mat_i, R, U, D, V);

		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = rest_shape_i[ne];
			Coord rest_pos_j = np_j.pos;

			Coord new_rest_pos_j = rest_pos_i + R*(rest_pos_j - rest_pos_i);
			np_j.pos = new_rest_pos_j;
			rest_shape_i[ne] = np_j;
		}
	}


	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::rotateRestShape()
	{
		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		EM_RotateRestShape <Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (
			this->inPosition()->getData(),
			m_bYield,
			this->inRestShape()->getData(),
			this->inHorizon()->getData());
		cuSynchronize();
	}

	template<typename TDataType>
	bool ElastoplasticityModule<TDataType>::initializeImpl()
	{
		m_invF.resize(this->inPosition()->getElementCount());
		m_yiled_I1.resize(this->inPosition()->getElementCount());
		m_yield_J2.resize(this->inPosition()->getElementCount());
		m_I1.resize(this->inPosition()->getElementCount());
		m_bYield.resize(this->inPosition()->getElementCount());

		m_bYield.reset();

		m_pbdModule = std::make_shared<DensityPBD<TDataType>>();
		this->inTimeStep()->connect(m_pbdModule->inTimeStep());
		this->inHorizon()->connect(m_pbdModule->varSmoothingLength());
		this->inPosition()->connect(m_pbdModule->inPosition());
		this->inVelocity()->connect(m_pbdModule->inVelocity());
		this->inNeighborIds()->connect(m_pbdModule->inNeighborIds());
		m_pbdModule->initialize();

		m_pbdModule->setParent(this->getParent());


		return ElasticityModule<TDataType>::initializeImpl();
	}


	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::setCohesion(Real c)
	{
		m_c.setValue(c);
	}


	template<typename TDataType>
	void ElastoplasticityModule<TDataType>::setFrictionAngle(Real phi)
	{
		m_phi.setValue(phi/180);
	}

	DEFINE_CLASS(ElastoplasticityModule);
}