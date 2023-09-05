#include "Multiply.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(MultiplyRealAndReal, TDataType);

	template<typename TDataType>
	void MultiplyRealAndReal<TDataType>::compute()
	{
		this->outO()->setValue(this->inA()->getValue() * this->inB()->getValue());
	}

	template class MultiplyRealAndReal<DataType3f>;
	template class MultiplyRealAndReal<DataType3d>;
}