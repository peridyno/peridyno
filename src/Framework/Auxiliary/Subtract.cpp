#include "Subtract.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(SubtractRealAndReal, TDataType);

	template<typename TDataType>
	void SubtractRealAndReal<TDataType>::compute()
	{
		this->outO()->setValue(this->inA()->getValue() - this->inB()->getValue());
	}

	template class SubtractRealAndReal<DataType3f>;
	template class SubtractRealAndReal<DataType3d>;
}