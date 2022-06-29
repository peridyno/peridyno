#include "SurfacePickerNode.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SurfacePickerNode, TDataType)

	template<typename TDataType>
	SurfacePickerNode<TDataType>::SurfacePickerNode(std::string name)
		:Node(name)
	{
		auto mouseInteractor=std::make_shared<SurfaceIteraction<TDataType>>();
		this->inInTopology()->connect(mouseInteractor->inInitialTriangleSet());

		this->stateSelectedTopology()->connect(mouseInteractor->outSelectedTriangleSet());
		this->stateOtherTopology()->connect(mouseInteractor->outOtherTriangleSet());
		mouseInteractor->setUpdateAlways(true);
		this->stateMouseInteractor()->setDataPtr(mouseInteractor);
		this->graphicsPipeline()->pushModule(mouseInteractor);
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
		this->stateSelectedTopology()->getDataPtr()->getTriangles().resize(0);
		this->stateOtherTopology()->getDataPtr()->copyFrom(this->inInTopology()->getData());
	}

	DEFINE_CLASS(SurfacePickerNode);
}