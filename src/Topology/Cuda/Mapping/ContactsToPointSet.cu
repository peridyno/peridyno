#include "ContactsToPointSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ContactsToPointSet, TDataType);

	template<typename TDataType>
	ContactsToPointSet<TDataType>::ContactsToPointSet()
		: TopologyMapping()
	{
	}

	template<typename Coord>
	__global__ void SetupContactPoints(
		DArray<Coord> vertices,
		DArray<TContactPair<Real>> contacts)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contacts.size()) return;

		auto contact = contacts[tId];
		Coord v0 = contact.pos1;

		vertices[tId] = v0;
	}

	template<typename TDataType>
	bool ContactsToPointSet<TDataType>::apply()
	{
		if (this->outPointSet()->isEmpty())
		{
			this->outPointSet()->allocate();
		}

		auto outSet = this->outPointSet()->getDataPtr();

		if (this->inContacts()->isEmpty())
		{
			outSet->clear();
			return true;
		}

		auto& inContacts = this->inContacts()->getData();

		auto& vertices = outSet->getPoints();

		uint contactNum = inContacts.size();
		vertices.resize(contactNum);

		cuExecute(contactNum,
			SetupContactPoints,
			vertices,
			inContacts);

		return true;
	}

	template<typename TDataType>
	bool ContactsToPointSet<TDataType>::validateInputs()
	{
		return true;
	}

	DEFINE_CLASS(ContactsToPointSet);
}