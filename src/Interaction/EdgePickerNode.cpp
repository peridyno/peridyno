#include "EdgePickerNode.h"
#include <iostream>

namespace dyno
{
	IMPLEMENT_TCLASS(EdgePickerNode, TDataType)

		template<typename TDataType>
	EdgePickerNode<TDataType>::EdgePickerNode(std::string name)
		:Node(name)
	{
		auto mouseInteractor = std::make_shared<EdgeIteraction<TDataType>>();
		this->inInTopology()->connect(mouseInteractor->inInitialTriangleSet());

		this->stateSelectedTopology()->connect(mouseInteractor->outSelectedEdgeSet());
		this->stateOtherTopology()->connect(mouseInteractor->outOtherEdgeSet());

		mouseInteractor->setUpdateAlways(true);
		this->stateMouseInteractor()->setDataPtr(mouseInteractor);

		this->graphicsPipeline()->pushModule(mouseInteractor);
	}

	template<typename TDataType>
	EdgePickerNode<TDataType>::~EdgePickerNode()
	{
	}

	template<typename TDataType>
	void EdgePickerNode<TDataType>::resetStates()
	{
		this->inInTopology()->getDataPtr()->updateEdges();
		this->stateOtherTopology()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
		this->stateSelectedTopology()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
		this->stateOtherTopology()->getDataPtr()->copyFrom(this->inInTopology()->getData());
	}

	DEFINE_CLASS(EdgePickerNode);
}