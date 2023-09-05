#include "Divide.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(DivideRealAndReal, TDataType);

	template<typename TDataType>
	void DivideRealAndReal<TDataType>::compute()
	{
		this->outO()->setValue(this->inA()->getValue() / this->inB()->getValue());
	}

	template class DivideRealAndReal<DataType3f>;
	template class DivideRealAndReal<DataType3d>;
}