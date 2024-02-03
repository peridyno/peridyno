#include "Extract.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ExtractEdgeSetFromPolygonSet, TDataType)

	template<typename TDataType>
	ExtractEdgeSetFromPolygonSet<TDataType>::ExtractEdgeSetFromPolygonSet()
		: TopologyMapping()
	{
		auto ts = std::make_shared<EdgeSet<TDataType>>();
		this->outEdgeSet()->setDataPtr(ts);
	}

	template<typename TDataType>
	ExtractEdgeSetFromPolygonSet<TDataType>::~ExtractEdgeSetFromPolygonSet()
	{
	}

	template<typename TDataType>
	bool ExtractEdgeSetFromPolygonSet<TDataType>::apply()
	{
		auto es = this->outEdgeSet()->getDataPtr();

		auto ps = this->inPolygonSet()->getDataPtr();

		ps->extractEdgeSet(*es);

		return true;
	}

	DEFINE_CLASS(ExtractEdgeSetFromPolygonSet);


	IMPLEMENT_TCLASS(ExtractTriangleSetFromPolygonSet, TDataType)

		template<typename TDataType>
	ExtractTriangleSetFromPolygonSet<TDataType>::ExtractTriangleSetFromPolygonSet()
		: TopologyMapping()
	{
		auto ts = std::make_shared<TriangleSet<TDataType>>();
		this->outTriangleSet()->setDataPtr(ts);
	}

	template<typename TDataType>
	ExtractTriangleSetFromPolygonSet<TDataType>::~ExtractTriangleSetFromPolygonSet()
	{
	}

	template<typename TDataType>
	bool ExtractTriangleSetFromPolygonSet<TDataType>::apply()
	{
		auto ts = this->outTriangleSet()->getDataPtr();

		auto ps = this->inPolygonSet()->getDataPtr();

		ps->turnIntoTriangleSet(*ts);

		return true;
	}

	DEFINE_CLASS(ExtractTriangleSetFromPolygonSet);
}