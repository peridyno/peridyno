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
		this->stateInTopology()->connect(mouseInteractor->inInitialTriangleSet());

		this->stateSelectedTopology()->connect(mouseInteractor->outSelectedEdgeSet());
		this->stateOtherTopology()->connect(mouseInteractor->outOtherEdgeSet());
		this->stateMouseInteractor()->setDataPtr(mouseInteractor);

		this->animationPipeline()->pushModule(mouseInteractor);
	}

	template<typename TDataType>
	EdgePickerNode<TDataType>::~EdgePickerNode()
	{
	}

	template<typename TDataType>
	void EdgePickerNode<TDataType>::resetStates()
	{
		this->stateInTopology()->getDataPtr()->updateEdges();
		this->stateOtherTopology()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
		this->stateSelectedTopology()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
		this->stateOtherTopology()->getDataPtr()->copyFrom(this->stateInTopology()->getData());
	}

	DEFINE_CLASS(EdgePickerNode);
}