#pragma once
#include "Collision/CollisionData.h"

#include "Module/ComputeModule.h"

namespace dyno
{
	template<typename TDataType>
	class ContactsUnion : public ComputeModule
	{
		DECLARE_CLASS_1(ContactsUnion, TDataType)
	public:
		typedef typename TContactPair<Real> ContactPair;

		ContactsUnion() {};
		~ContactsUnion() override {};

		void compute() override;

	public:
		DEF_ARRAY_IN(ContactPair, ContactsA, DeviceType::GPU, "");
		DEF_ARRAY_IN(ContactPair, ContactsB, DeviceType::GPU, "");

		DEF_ARRAY_OUT(ContactPair, Contacts, DeviceType::GPU, "");
	};
}