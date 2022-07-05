#include "SurfacePickerNode.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SurfacePickerNode, TDataType)

	template<typename TDataType>
	SurfacePickerNode<TDataType>::SurfacePickerNode(std::string name)
		:Node(name)
	{
		auto mouseInteractor=std::make_shared<SurfaceInteraction<TDataType>>();
		this->inInTopology()->connect(mouseInteractor->inInitialTriangleSet());
		this->stateSelectedTopology()->connect(mouseInteractor->outSelectedTriangleSet());
		this->stateOtherTopology()->connect(mouseInteractor->outOtherTriangleSet());

		mouseInteractor->setUpdateAlways(true);
		this->mouseInteractor = mouseInteractor;
		this->graphicsPipeline()->pushModule(mouseInteractor);

		this->stateSelectedTopology()->promoteOuput();
		this->stateOtherTopology()->promoteOuput();
	}

	template<typename TDataType>
	SurfacePickerNode<TDataType>::~SurfacePickerNode()
	{
	}

	template<typename TDataType>
	std::string SurfacePickerNode<TDataType>::getNodeType()
	{
		return "Interaction";
	}

	template<typename TDataType>
	void SurfacePickerNode<TDataType>::resetStates()
	{
		this->stateOtherTopology()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->stateSelectedTopology()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->stateSelectedTopology()->getDataPtr()->getTriangles().resize(0);
		this->stateOtherTopology()->getDataPtr()->copyFrom(this->inInTopology()->getData());
	}

	DEFINE_CLASS(SurfacePickerNode);
}