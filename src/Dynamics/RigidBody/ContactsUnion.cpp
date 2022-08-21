#include "ContactsUnion.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(ContactsUnion, TDataType)

	template<typename TDataType>
	ContactsUnion<TDataType>::ContactsUnion()
	{
		this->inContactsA()->tagOptional(true);
		this->inContactsB()->tagOptional(true);
	}

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

		DArray<ContactPair> outData;
		if (this->outContacts()->getDataPtr() != nullptr)
			outData = this->outContacts()->getData();

		if (inDataA != nullptr)
			outData.assign(*inDataA, inDataA->size());

		if (inDataB != nullptr)
			outData.assign(*inDataB, inDataB->size(), inDataA->size(), 0);
	}

	DEFINE_CLASS(ContactsUnion);
}