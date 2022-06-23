#include "PointPickerNode.h"
#include <iostream>

namespace dyno
{
	IMPLEMENT_TCLASS(PointPickerNode, TDataType)

		template<typename TDataType>
	PointPickerNode<TDataType>::PointPickerNode(std::string name)
		:Node(name)
	{
		auto mouseInteractor = std::make_shared<PointIteraction<TDataType>>();
		this->stateInTopology()->connect(mouseInteractor->inInitialTriangleSet());

		this->stateSelectedTopology()->connect(mouseInteractor->outSelectedPointSet());
		this->stateOtherTopology()->connect(mouseInteractor->outOtherPointSet());
		this->stateMouseInteractor()->setDataPtr(mouseInteractor);

		this->animationPipeline()->pushModule(mouseInteractor);
	}

	template<typename TDataType>
	PointPickerNode<TDataType>::~PointPickerNode()
	{
	}

	template<typename TDataType>
	void PointPickerNode<TDataType>::resetStates()
	{
		this->stateOtherTopology()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->stateSelectedTopology()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->stateOtherTopology()->getDataPtr()->copyFrom(this->stateInTopology()->getData());
	}

	DEFINE_CLASS(PointPickerNode);
}