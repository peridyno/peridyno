#include "Add.h"

namespace dyno
{
	IMPLEMENT_TCLASS(AddRealAndReal, TDataType);

	template<typename TDataType>
	void AddRealAndReal<TDataType>::compute()
	{
		this->outO()->setValue(this->inA()->getValue() + this->inB()->getValue());
	}

	template class AddRealAndReal<DataType3f>;
	template class AddRealAndReal<DataType3d>;
}