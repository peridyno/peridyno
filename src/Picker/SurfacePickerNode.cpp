#include "SurfacePickerNode.h"
#include "CustomMouseInteraction.h"
#include "MouseIntersect.h"
#include <iostream>

namespace dyno
{
	IMPLEMENT_TCLASS(SurfacePickerNode, TDataType)

	template<typename TDataType>
	SurfacePickerNode<TDataType>::SurfacePickerNode(std::string name)
		:Node(name)
	{
		auto mouseInteractor=std::make_shared<CustomMouseIteraction<TDataType>>();
		this->stateInTopology()->connect(mouseInteractor->inInitialTriangleSet());

		this->stateSelectedTopology()->connect(mouseInteractor->outSelectedTriangleSet());
		this->stateOtherTopology()->connect(mouseInteractor->outOtherTriangleSet());
		this->stateMouseInteractor()->setDataPtr(mouseInteractor);

		this->animationPipeline()->pushModule(mouseInteractor);
	}

	template<typename TDataType>
	SurfacePickerNode<TDataType>::~SurfacePickerNode()
	{
	}

	template<typename TDataType>
	void SurfacePickerNode<TDataType>::resetStates()
	{
		this->stateOtherTopology()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->stateSelectedTopology()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->stateOtherTopology()->getDataPtr()->copyFrom(this->stateInTopology()->getData());
		this->stateSelectedTopology()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
	}

	DEFINE_CLASS(SurfacePickerNode);
}