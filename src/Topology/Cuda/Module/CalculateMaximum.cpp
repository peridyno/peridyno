#include "CalculateMaximum.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(CalculateMaximum, TDataType)

	template<typename TDataType>
	void CalculateMaximum<TDataType>::compute()
	{
		auto& inData = this->inScalarArray()->getData();

		auto maxV = mReduce.maximum(inData.begin(), inData.size());

		this->outScalar()->setValue(maxV);
	}

	DEFINE_CLASS(CalculateMaximum);
}