#include "ContactsUnion.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(ContactsUnion, TDataType)

	template<typename TDataType>
	void ContactsUnion<TDataType>::compute()
	{
		auto inDataA = this->inContactsA()->getDataPtr();
		auto inDataB = this->inContactsB()->getDataPtr();
		
		uint total_size = 0;
		if (inDataA != nullptr)
			total_size += inDataA->size();

		if (inDataB != nullptr)
			total_size += inDataB->size();

		if (this->outContacts()->size() != total_size)
			this->outContacts()->resize(total_size);

		auto& outData = this->outContacts()->getData();

		if (inDataA != nullptr)
			outData.assign(*inDataA, inDataA->size());

		if (inDataB != nullptr)
			outData.assign(*inDataB, inDataB->size(), inDataA->size(), 0);
	}

	template<typename TDataType>
	bool ContactsUnion<TDataType>::validateInputs()
	{
		bool ret = this->inContactsA()->isEmpty() && this->inContactsB()->isEmpty();

		return !ret;
	}

	DEFINE_CLASS(ContactsUnion);
}