#include "ColorMapping.h"
#include "Algorithm/SimpleMath.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(ColorMapping, TDataType)

	template <typename Real>
	__global__ void CM_MapJetColor(
		DArray<Vec3f> color,
		DArray<Real> v,
		Real vmin,
		Real vmax)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= color.size()) return;
		
		Real x = clamp((v[tId] - vmin) / (vmax - vmin), Real(0), Real(1));
		Real r = clamp(Real(-4 * abs(x - 0.75) + 1.5), Real(0), Real(1));
		Real g = clamp(Real(-4 * abs(x - 0.50) + 1.5), Real(0), Real(1));
		Real b = clamp(Real(-4 * abs(x - 0.25) + 1.5), Real(0), Real(1));
		color[tId] = Vec3f(r, g, b);
	}

	template <typename Real>
	__global__ void CM_MapHeatColor(
		DArray<Vec3f> color,
		DArray<Real> v,
		Real vmin,
		Real vmax)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= color.size()) return;

		Real x = clamp((v[tId] - vmin) / (vmax - vmin), Real(0), Real(1));
		Real r = clamp(Real(-4 * abs(x - 0.75) + 2), Real(0), Real(1));
		Real g = clamp(Real(-4 * abs(x - 0.50) + 2), Real(0), Real(1));
		Real b = clamp(Real(-4 * abs(x) + 2), Real(0), Real(1));
		color[tId] = Vec3f(r, g, b);
	}

	template<typename TDataType>
	void ColorMapping<TDataType>::compute()
	{
		auto& inData = this->inScalar()->getData();

		int num = inData.size();

		if (this->outColor()->isEmpty())
		{
			this->outColor()->allocate();
		}

		auto& outData = this->outColor()->getData();
		if (outData.size() != num)
		{
			outData.resize(num);
		}

		if(this->varType()->getData() == ColorTable::Jet)
		{
			cuExecute(num,
				CM_MapJetColor,
				outData,
				inData,
				this->varMin()->getData(),
				this->varMax()->getData());
		}
		else if(this->varType()->getData() == ColorTable::Heat)
		{
			cuExecute(num,
				CM_MapHeatColor,
				outData,
				inData,
				this->varMin()->getData(),
				this->varMax()->getData());
		}
	}

	DEFINE_CLASS(ColorMapping);
}