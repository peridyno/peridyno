#include "ContactsToEdgeSet.h"

namespace dyno
{
	template<typename TDataType>
	ContactsToEdgeSet<TDataType>::ContactsToEdgeSet()
		: TopologyMapping()
	{
	}

	template<typename Coord, typename Edge>
	__global__ void SetupContactInfo(
		DArray<Coord> vertices,
		DArray<Edge> indices,
		DArray<TContactPair<Real>> contacts, 
		Real scale)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= contacts.size()) return;

		auto contact = contacts[tId];
		Coord v0 = contact.pos1;
		Coord v1 = contact.pos1 + contact.normal1 * scale;

		vertices[2 * tId] = v0;
		vertices[2 * tId + 1] = v1;
		indices[tId] = Edge(2 * tId, 2 * tId + 1);
	}

	template<typename TDataType>
	bool ContactsToEdgeSet<TDataType>::apply()
	{
		if (this->outEdgeSet()->isEmpty())
		{
			this->outEdgeSet()->allocate();
		}

		auto& inContacts = this->inContacts()->getData();
		auto outSet = this->outEdgeSet()->getDataPtr();

		auto& vertices = outSet->getPoints();
		auto& indices = outSet->getEdges();

		uint contactNum = inContacts.size();
		vertices.resize(2 * contactNum);
		indices.resize(contactNum);

		cuExecute(contactNum,
			SetupContactInfo,
			vertices,
			indices,
			inContacts,
			this->varScale()->getData());

		return true;
	}

	DEFINE_CLASS(ContactsToEdgeSet);
}