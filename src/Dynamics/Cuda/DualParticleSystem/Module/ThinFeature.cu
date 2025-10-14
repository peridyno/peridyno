#include "ThinFeature.h"
#include "Matrix/MatrixFunc.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ThinFeature, TDataType)

	template<typename TDataType>
	ThinFeature<TDataType>::ThinFeature()
		: ParticleApproximation<TDataType>()
	{
		mSummation = std::make_shared<SummationDensity<TDataType>>();

		this->inSmoothingLength()->connect(mSummation->inSmoothingLength());
		this->inSamplingDistance()->connect(mSummation->inSamplingDistance());
		this->inPosition()->connect(mSummation->inPosition());
		this->inNeighborIds()->connect(mSummation->inNeighborIds());

		mSummation->outDensity()->connect(this->outDensity());
		this->varKernelType()->getDataPtr()->setCurrentKey(1);
	}

	template<typename TDataType>
	ThinFeature<TDataType>::~ThinFeature()
	{
		mDistributMat.clear();
		mEigens.clear();
	}

	template <typename Real, typename Coord, typename Matrix, typename Kernel>
	__global__ void SphThinFeatrue_DistriMatCompute(
		DArray<Matrix> distriMatrix,
		DArray<Coord> Eigens,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		Real rho_0,
		Real smoothingLength,
		Real samplingDistance,
		Real mass,
		Kernel weight,
		Real scale)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		Real total_weight = 0.0f; 
		Matrix tm(0.0f);

		Real wij = 1.0f;

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Coord& X_j = posArr[j];
			Coord& X_i = posArr[pId];

			Coord xij = (X_j - X_i) / smoothingLength;
			Real rij = (X_j - X_i).norm();

			if ((rij > EPSILON)||(pId != j))
			{
				//Real wij = weight(rij, smoothingLength, scale);
				total_weight += wij;
				
				tm(0, 0) += xij[0] * xij[0] * wij;	tm(0, 1) += xij[0] * xij[1] * wij;	tm(0, 2) += xij[0] * xij[2] * wij;
				tm(1, 0) += xij[1] * xij[0] * wij;	tm(1, 1) += xij[1] * xij[1] * wij;	tm(1, 2) += xij[1] * xij[2] * wij;
				tm(2, 0) += xij[2] * xij[0] * wij;	tm(2, 1) += xij[2] * xij[1] * wij;	tm(2, 2) += xij[2] * xij[2] * wij;
			}
		}

		if (total_weight > EPSILON)
		{
			tm *= (1.0 / total_weight);
		}
		else
		{
			tm = Matrix::identityMatrix();
		}

		distriMatrix[pId] = tm;

		Matrix R, U, D, V;
		polarDecomposition(tm, R, U, D, V);

		Eigens[pId][0] = D(0, 0);
		Eigens[pId][1] = D(1, 1);
		Eigens[pId][2] = D(2, 2);
	}


	template <typename Real, typename Coord>
	__global__ void SphThinFeatrue_ThinSheetJudge(
		DArray<Coord> Eigens,
		DArray<bool> thinSheetFlag,
		DArray<Real> thinFeature,
		Real threshold)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Eigens.size()) return;

		Coord& Eigen_i = Eigens[pId];

		if ((Eigen_i[0] < threshold) || (Eigen_i[1] < threshold) || (Eigen_i[2] < threshold))
		{
			thinSheetFlag[pId] = true;
		}
		else
		{
			thinSheetFlag[pId] = false;
		}
	}

	template<typename TDataType>
	void ThinFeature<TDataType>::resizeArray(int num)
	{
		if (mDistributMat.size() != num)
		{
			mDistributMat.resize(num);
		}

		if (mEigens.size() != num)
		{
			mEigens.resize(num);
		}

		if (this->outThinSheet()->size() != num)
		{
			this->outThinSheet()->allocate();
			this->outThinSheet()->resize(num);
		}

		if (this->outThinFeature()->size() != num)
		{
			this->outThinFeature()->allocate();
			this->outThinFeature()->resize(num);
		}
	}


	template<typename TDataType>
	void ThinFeature<TDataType>::compute()
	{
		//std::cout << "Feature!" << std::endl;
		int num = this->inPosition()->size();

		this->resizeArray(num);

		mSummation->varRestDensity()->setValue(this->varRestDensity()->getValue());
		mSummation->varKernelType()->setCurrentKey(this->varKernelType()->currentKey());
		mSummation->update();

		Real mass = mSummation->getParticleMass();

		cuZerothOrder(num, this->varKernelType()->getDataPtr()->currentKey(), this->mScalingFactor,
			SphThinFeatrue_DistriMatCompute,
			mDistributMat,
			mEigens,
			this->inPosition()->getData(),
			this->inNeighborIds()->getData(),
			this->varRestDensity()->getValue(),
			this->inSmoothingLength()->getValue(),
			this->inSamplingDistance()->getValue(),
			mass);

		cuExecute(num, 
			SphThinFeatrue_ThinSheetJudge,
			mEigens,
			this->outThinSheet()->getData(),
			this->outThinFeature()->getData(),
			this->varThreshold()->getValue());
	}


	DEFINE_CLASS(ThinFeature);
}