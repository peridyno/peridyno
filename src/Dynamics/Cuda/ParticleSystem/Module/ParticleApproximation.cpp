#include "ParticleApproximation.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ParticleApproximation, TDataType)

	template<typename TDataType>
	ParticleApproximation<TDataType>::ParticleApproximation()
		: ComputeModule()
	{
// 		this->inSmoothingLength()->setValue(Real(0.011));
// 		this->inSamplingDistance()->setValue(Real(0.005));

		auto callback = std::make_shared<FCallBackFunc>(
			std::bind(&ParticleApproximation<TDataType>::calculateScalingFactor, this));

		this->varKernelType()->attach(callback);
		this->inSmoothingLength()->attach(callback);
		this->inSamplingDistance()->attach(callback);

		//Should be called after above four parameters are all set, this function will recalculate m_factor
		//calculateScalingFactor();
	}

	template<typename TDataType>
	ParticleApproximation<TDataType>::~ParticleApproximation()
	{
	}

	template<typename TDataType>
	void ParticleApproximation<TDataType>::calculateScalingFactor()
	{
		Real d = this->inSamplingDistance()->getValue();
		Real H = this->inSmoothingLength()->getValue();

		Real V = d * d*d;

		Kernel<Real>* kern;
		switch (this->varKernelType()->currentKey())
		{
		case KT_Spiky:
			kern = new SpikyKernel<Real>();
			break;
		case KT_Smooth:
			kern = new SmoothKernel<Real>();
			break;
		default:
			break;
		}

		Real total_weight(0);
		int half_res = (int)(H / d + 1);
		for (int i = -half_res; i <= half_res; i++)
			for (int j = -half_res; j <= half_res; j++)
				for (int k = -half_res; k <= half_res; k++)
				{
					Real x = i * d;
					Real y = j * d;
					Real z = k * d;
					Real r = sqrt(x * x + y * y + z * z);
					total_weight += V * kern->Weight(r, H);
				}

		mScalingFactor = Real(1) / total_weight;

		delete kern;
	}

	DEFINE_CLASS(ParticleApproximation);
}