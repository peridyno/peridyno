#include "MergeTriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	MergeTriangleSet<TDataType>::MergeTriangleSet()
		: Node()
	{
		this->allowExported(false);

		auto ts = std::make_shared<TriangleSet<TDataType>>();
		this->stateTriangleSet()->setDataPtr(ts);

		this->stateTriangleSet()->promoteOuput();
	}

	template<typename TDataType>
	MergeTriangleSet<TDataType>::~MergeTriangleSet()
	{
	}

	template<typename TDataType>
	void MergeTriangleSet<TDataType>::resetStates()
	{
		auto first = this->inFirst()->getDataPtr();
		auto second = this->inSecond()->getDataPtr();

		auto topo = this->stateTriangleSet()->getDataPtr();

		topo->copyFrom(*first->merge(*second));
	}

	DEFINE_CLASS(MergeTriangleSet);
}