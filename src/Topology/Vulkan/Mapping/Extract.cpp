#include "Extract.h"

namespace dyno
{
	/**
	 * ExtractEdgeSetFromPolygonSet
	 */
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

	/**
	 * ExtractTriangleSetFromPolygonSet
	 */
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

	/**
	 * ExtractQaudSetFromPolygonSet
	 */
	IMPLEMENT_TCLASS(ExtractQaudSetFromPolygonSet, TDataType)

	template<typename TDataType>
	ExtractQaudSetFromPolygonSet<TDataType>::ExtractQaudSetFromPolygonSet()
		: TopologyMapping()
	{
		auto ts = std::make_shared<QuadSet<TDataType>>();
		this->outQuadSet()->setDataPtr(ts);
	}

	template<typename TDataType>
	ExtractQaudSetFromPolygonSet<TDataType>::~ExtractQaudSetFromPolygonSet()
	{
	}

	template<typename TDataType>
	bool ExtractQaudSetFromPolygonSet<TDataType>::apply()
	{
		auto qs = this->outQuadSet()->getDataPtr();

		auto ps = this->inPolygonSet()->getDataPtr();

		ps->extractQuadSet(*qs);

		return true;
	}

	DEFINE_CLASS(ExtractQaudSetFromPolygonSet);
}