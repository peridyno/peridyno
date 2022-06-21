#include "StaticTriangularMesh.h"

#include "Topology/TriangleSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(StaticTriangularMesh, TDataType)

	template<typename TDataType>
	StaticTriangularMesh<TDataType>::StaticTriangularMesh()
		: Node()
	{
		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		this->stateTopology()->setDataPtr(triSet);

		this->outTriangleSet()->setDataPtr(triSet);
	}

	template<typename TDataType>
	void StaticTriangularMesh<TDataType>::resetStates()
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTopology()->getDataPtr());

		triSet->loadObjFile(this->varFileName()->getDataPtr()->string());

		triSet->scale(this->varScale()->getData());
		triSet->translate(this->varLocation()->getData());

		Node::resetStates();
	}

	DEFINE_CLASS(StaticTriangularMesh);
}