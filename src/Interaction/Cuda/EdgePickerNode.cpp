#include "EdgePickerNode.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"

namespace dyno
{
	IMPLEMENT_TCLASS(EdgePickerNode, TDataType)

	template<typename TDataType>
	EdgePickerNode<TDataType>::EdgePickerNode()
		:Node()
	{
		auto edgeInteractor = std::make_shared<EdgeInteraction<TDataType>>();
		auto pointInteractor = std::make_shared<PointInteraction<TDataType>>();
		this->inTopology()->connect(edgeInteractor->inInitialEdgeSet());
		this->inTopology()->connect(pointInteractor->inInitialPointSet());

		this->varInteractionRadius()->connect(edgeInteractor->varInteractionRadius());
		this->varInteractionRadius()->connect(pointInteractor->varInteractionRadius());

		this->stateEdgeIndex()->connect(edgeInteractor->outEdgeIndex());
		this->statePointIndex()->connect(pointInteractor->outPointIndex());

		this->varToggleIndexOutput()->connect(edgeInteractor->varToggleIndexOutput());
		this->varToggleIndexOutput()->connect(pointInteractor->varToggleIndexOutput());

		this->edgeInteractor = edgeInteractor;
		this->pointInteractor = pointInteractor;

		this->graphicsPipeline()->pushModule(edgeInteractor);
		this->graphicsPipeline()->pushModule(pointInteractor);

		auto edgeRender1 = std::make_shared<GLWireframeVisualModule>();
		this->varEdgeSelectedSize()->connect(edgeRender1->varRadius());
		edgeRender1->setColor(Color(0.8f, 0.0f, 0.0f));
		this->edgeInteractor->outSelectedEdgeSet()->connect(edgeRender1->inEdgeSet());
		this->graphicsPipeline()->pushModule(edgeRender1);

		auto edgeRender2 = std::make_shared<GLWireframeVisualModule>();
		this->varEdgeOtherSize()->connect(edgeRender2->varRadius());
		edgeRender2->setColor(Color(0.0f, 0.0f, 0.0f));
		this->edgeInteractor->outOtherEdgeSet()->connect(edgeRender2->inEdgeSet());
		this->graphicsPipeline()->pushModule(edgeRender2);

		auto pointRender1 = std::make_shared<GLPointVisualModule>();
		this->varPointSelectedSize()->connect(pointRender1->varPointSize());
		pointRender1->setColor(Color(1.0f, 0.0f, 0.0f));
		this->pointInteractor->outSelectedPointSet()->connect(pointRender1->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender1);

		auto pointRender2 = std::make_shared<GLPointVisualModule>();
		this->varPointOtherSize()->connect(pointRender2->varPointSize());
		pointRender2->setColor(Color(0.0f, 0.0f, 1.0f));
		this->pointInteractor->outOtherPointSet()->connect(pointRender2->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender2);

		this->varInteractionRadius()->setRange(0.001f, 0.2f);
		this->varInteractionRadius()->setValue(0.01f);
		this->varPointSelectedSize()->setRange(0.0f, 0.1f);
		this->varPointOtherSize()->setRange(0.0f, 0.1f);

		auto callback1 = std::make_shared<FCallBackFunc>(std::bind(&EdgePickerNode<TDataType>::changePickingElementType, this));

		this->varPickingElementType()->attach(callback1);

		auto callback2 = std::make_shared<FCallBackFunc>(std::bind(&EdgePickerNode<TDataType>::changePickingType, this));

		this->varPickingType()->attach(callback2);

		auto callback3 = std::make_shared<FCallBackFunc>(std::bind(&EdgePickerNode<TDataType>::changeMultiSelectionType, this));

		this->varMultiSelectionType()->attach(callback3);

		this->edgeInteractor->outEdgeIndex()->allocate();
		this->pointInteractor->outPointIndex()->allocate();
	}

	template<typename TDataType>
	EdgePickerNode<TDataType>::~EdgePickerNode()
	{
	}

	template<typename TDataType>
	std::string EdgePickerNode<TDataType>::getNodeType()
	{
		return "Interaction";
	}

	template<typename TDataType>
	void EdgePickerNode<TDataType>::resetStates()
	{
		//		this->inTopology()->getDataPtr()->update();
		this->edgeInteractor->outEdgeIndex()->allocate();
		this->pointInteractor->outPointIndex()->allocate();

		this->edgeInteractor->outOtherEdgeSet()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
		this->edgeInteractor->outSelectedEdgeSet()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
		this->edgeInteractor->outOtherEdgeSet()->getDataPtr()->copyFrom(this->inTopology()->getData());

		this->pointInteractor->outOtherPointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->pointInteractor->outSelectedPointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->pointInteractor->outOtherPointSet()->getDataPtr()->copyFrom(this->inTopology()->getData());
	}

	template<typename TDataType>
	void EdgePickerNode<TDataType>::changePickingElementType()
	{
		if (this->varPickingElementType()->getValue() == PickingElementTypeSelection::Edge)
		{
			this->edgeInteractor->varTogglePicker()->setValue(true);
			this->pointInteractor->varTogglePicker()->setValue(false);
		}
		else if (this->varPickingElementType()->getValue() == PickingElementTypeSelection::Point)
		{
			this->edgeInteractor->varTogglePicker()->setValue(false);
			this->pointInteractor->varTogglePicker()->setValue(true);
		}
		else if (this->varPickingElementType()->getValue() == PickingElementTypeSelection::All)
		{
			this->edgeInteractor->varTogglePicker()->setValue(true);
			this->pointInteractor->varTogglePicker()->setValue(true);
		}
		resetStates();
	}

	template<typename TDataType>
	void EdgePickerNode<TDataType>::changePickingType()
	{
		if (this->varPickingType()->getValue() == PickingTypeSelection::Click)
		{
			this->edgeInteractor->varEdgePickingType()->getDataPtr()->setCurrentKey(0);
			this->pointInteractor->varPointPickingType()->getDataPtr()->setCurrentKey(0);
		}
		else if (this->varPickingType()->getValue() == PickingTypeSelection::Drag)
		{
			this->edgeInteractor->varEdgePickingType()->getDataPtr()->setCurrentKey(1);
			this->pointInteractor->varPointPickingType()->getDataPtr()->setCurrentKey(1);
		}
		else if (this->varPickingType()->getValue() == PickingTypeSelection::Both)
		{
			this->edgeInteractor->varEdgePickingType()->getDataPtr()->setCurrentKey(2);
			this->pointInteractor->varPointPickingType()->getDataPtr()->setCurrentKey(2);
		}
		resetStates();
	}

	template<typename TDataType>
	void EdgePickerNode<TDataType>::changeMultiSelectionType()
	{
		if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
		{
			this->edgeInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(0);
			this->pointInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(0);
		}
		else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
		{
			this->edgeInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(1);
			this->pointInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(1);
		}
		else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
		{
			this->edgeInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(2);
			this->pointInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(2);
		}
	}

	DEFINE_CLASS(EdgePickerNode);
}