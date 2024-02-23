#include "CalculateMaxPoint.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(CalculateMaxPoint, TDataType)

	template<typename TDataType>
	void CalculateMaxPoint<TDataType>::compute()
	{
		auto& pointData = this->inpointSet()->getData();
		auto& pData = pointData.getPoints();
		if(pData.size()) {
			auto maxV = mReduce.maximum(pData.begin(), pData.size());
			this->outScalar()->setValue(maxV);
		}
	}

	DEFINE_CLASS(CalculateMaxPoint);
}