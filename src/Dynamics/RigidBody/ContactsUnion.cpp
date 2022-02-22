#include "ContactsUnion.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(ContactsUnion, TDataType)

	template<typename TDataType>
	void ContactsUnion<TDataType>::compute()
	{
		auto& inDataA = this->inContactsA()->getData();
		auto& inDataB = this->inContactsB()->getData();
		
		uint total_size = inDataA.size() + inDataB.size();

		if (this->outContacts()->isEmpty())
		{
			this->outContacts()->allocate();
		}

		auto& outData = this->outContacts()->getData();
		if (outData.size() != total_size)
		{
			outData.resize(total_size);
		}

		outData.assign(inDataA, inDataA.size());
		outData.assign(inDataB, inDataB.size(), inDataA.size(), 0);
	}

	DEFINE_CLASS(ContactsUnion);
}