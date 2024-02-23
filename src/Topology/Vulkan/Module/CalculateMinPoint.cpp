#include "CalculateMinPoint.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(CalculateMinPoint, TDataType)

	template<typename TDataType>
	void CalculateMinPoint<TDataType>::compute()
	{
		auto& pointData = this->inpointSet()->getData();
		auto& pData = pointData.getPoints();
		if(pData.size()) {
			auto minV = mReduce.minimum(pData.begin(), pData.size());
			this->outScalar()->setValue(minV);
		}
	}

	DEFINE_CLASS(CalculateMinPoint);
}