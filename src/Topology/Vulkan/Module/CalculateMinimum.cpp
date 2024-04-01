#include "CalculateMinimum.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(CalculateMinimum, TDataType)

	template<typename TDataType>
	void CalculateMinimum<TDataType>::compute()
	{
		auto& inData = this->inScalarArray()->getData();

		auto minV = mReduce.minimum(inData.begin(), inData.size());

		this->outScalar()->setValue(minV);
	}

	DEFINE_CLASS(CalculateMinimum);
}