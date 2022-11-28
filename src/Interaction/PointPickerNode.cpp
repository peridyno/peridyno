#include "PointPickerNode.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"

namespace dyno
{
	IMPLEMENT_TCLASS(PointPickerNode, TDataType)

	template<typename TDataType>
	PointPickerNode<TDataType>::PointPickerNode(std::string name)
		:Node(name)
	{
		auto pointInteractor = std::make_shared<PointInteraction<TDataType>>();

		this->inTopology()->connect(pointInteractor->inInitialPointSet());

		this->varInterationRadius()->connect(pointInteractor->varInterationRadius());

		this->statePointIndex()->connect(pointInteractor->outPointIndex());

		this->pointInteractor = pointInteractor;

		this->graphicsPipeline()->pushModule(pointInteractor);

		auto pointRender1 = std::make_shared<GLPointVisualModule>();
		this->varPointSelectedSize()->connect(pointRender1->varPointSize());
		this->varSelectedPointColor()->connect(pointRender1->varBaseColor());
		this->pointInteractor->outSelectedPointSet()->connect(pointRender1->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender1);

		auto pointRender2 = std::make_shared<GLPointVisualModule>();
		this->varPointOtherSize()->connect(pointRender2->varPointSize());
		this->varOtherPointColor()->connect(pointRender2->varBaseColor());
		this->pointInteractor->outOtherPointSet()->connect(pointRender2->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender2);

		this->varInterationRadius()->setRange(0.001f , 0.2f);
		this->varInterationRadius()->setValue(0.01f);
		this->varPointSelectedSize()->setRange(0.0f, 0.1f);
		this->varPointOtherSize()->setRange(0.0f,0.1f);

		auto callback2 = std::make_shared<FCallBackFunc>(std::bind(&PointPickerNode<TDataType>::changePickingType, this));

		this->varPickingType()->attach(callback2);

		auto callback3 = std::make_shared<FCallBackFunc>(std::bind(&PointPickerNode<TDataType>::changeMultiSelectionType, this));

		this->varMultiSelectionType()->attach(callback3);
	}

	template<typename TDataType>
	PointPickerNode<TDataType>::~PointPickerNode()
	{
	}

	template<typename TDataType>
	std::string PointPickerNode<TDataType>::getNodeType()
	{
		return "Interaction";
	}

	template<typename TDataType>
	void PointPickerNode<TDataType>::resetStates()
	{
//		this->inTopology()->getDataPtr()->update();
		this->pointInteractor->outPointIndex()->allocate();

		this->pointInteractor->outOtherPointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->pointInteractor->outSelectedPointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->pointInteractor->outOtherPointSet()->getDataPtr()->copyFrom(this->inTopology()->getData());
	}

	template<typename TDataType>
	void PointPickerNode<TDataType>::changePickingType()
	{
		if (this->varPickingType()->getValue() == PickingTypeSelection::Click)
		{
			this->pointInteractor->varPointPickingType()->getDataPtr()->setCurrentKey(0);
		}
		else if (this->varPickingType()->getValue() == PickingTypeSelection::Drag)
		{
			this->pointInteractor->varPointPickingType()->getDataPtr()->setCurrentKey(1);
		}
		else if (this->varPickingType()->getValue() == PickingTypeSelection::Both)
		{
			this->pointInteractor->varPointPickingType()->getDataPtr()->setCurrentKey(2);
		}
		resetStates();
	}

	template<typename TDataType>
	void PointPickerNode<TDataType>::changeMultiSelectionType()
	{
		if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
		{
			this->pointInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(0);
		}
		else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
		{
			this->pointInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(1);
		}
		else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
		{
			this->pointInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(2);
		}
	}

	DEFINE_CLASS(PointPickerNode);
}