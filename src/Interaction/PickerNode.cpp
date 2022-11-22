#include "PickerNode.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"

namespace dyno
{
	IMPLEMENT_TCLASS(PickerNode, TDataType)

	template<typename TDataType>
	PickerNode<TDataType>::PickerNode(std::string name)
		:Node(name)
	{
		auto surfaceInteractor = std::make_shared<SurfaceInteraction<TDataType>>();
		auto edgeInteractor = std::make_shared<EdgeInteraction<TDataType>>();
		auto pointInteractor = std::make_shared<PointInteraction<TDataType>>();

		this->inTopology()->connect(surfaceInteractor->inInitialTriangleSet());
		this->inTopology()->connect(edgeInteractor->inInitialTriangleSet());
		this->inTopology()->connect(pointInteractor->inInitialTriangleSet());

		this->varFloodAngle()->connect(surfaceInteractor->varFloodAngle());
		this->varToggleFlood()->connect(surfaceInteractor->varToggleFlood());
		this->varToggleVisibleFilter()->connect(surfaceInteractor->varToggleVisibleFilter());

		this->varInterationRadius()->connect(surfaceInteractor->varInterationRadius());
		this->varInterationRadius()->connect(edgeInteractor->varInterationRadius());
		this->varInterationRadius()->connect(pointInteractor->varInterationRadius());

		this->stateTriangleIndex()->connect(surfaceInteractor->outTriangleIndex());
		this->stateEdgeIndex()->connect(edgeInteractor->outEdgeIndex());
		this->statePointIndex()->connect(pointInteractor->outPointIndex());

		this->surfaceInteractor = surfaceInteractor;
		this->edgeInteractor = edgeInteractor;
		this->pointInteractor = pointInteractor;

		this->graphicsPipeline()->pushModule(surfaceInteractor);
		this->graphicsPipeline()->pushModule(edgeInteractor);
		this->graphicsPipeline()->pushModule(pointInteractor);

		auto surfaceRender1 = std::make_shared<GLSurfaceVisualModule>();
		surfaceRender1->varAlpha()->setValue(0.95f);
		this->varSelectedTriangleColor()->connect(surfaceRender1->varBaseColor());
		this->surfaceInteractor->outSelectedTriangleSet()->connect(surfaceRender1->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfaceRender1);

		auto surfaceRender2 = std::make_shared<GLSurfaceVisualModule>();
		surfaceRender2->varAlpha()->setValue(0.85f);
		this->varOtherTriangleColor()->connect(surfaceRender2->varBaseColor());
		this->surfaceInteractor->outOtherTriangleSet()->connect(surfaceRender2->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfaceRender2);

		auto edgeRender1 = std::make_shared<GLWireframeVisualModule>();
		this->varSelectedEdgeColor()->connect(edgeRender1->varBaseColor());
		this->edgeInteractor->outSelectedEdgeSet()->connect(edgeRender1->inEdgeSet());
		this->graphicsPipeline()->pushModule(edgeRender1);

		auto edgeRender2 = std::make_shared<GLWireframeVisualModule>();
		this->varOtherEdgeColor()->connect(edgeRender2->varBaseColor());
		this->edgeInteractor->outOtherEdgeSet()->connect(edgeRender2->inEdgeSet());
		this->graphicsPipeline()->pushModule(edgeRender2);

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
		this->varFloodAngle()->setRange(0.0f, M_PI);

		auto callback1 = std::make_shared<FCallBackFunc>(std::bind(&PickerNode<TDataType>::changePickingElementType, this));

		this->varPickingElementType()->attach(callback1);

		auto callback2 = std::make_shared<FCallBackFunc>(std::bind(&PickerNode<TDataType>::changePickingType, this));

		this->varPickingType()->attach(callback2);

		auto callback3 = std::make_shared<FCallBackFunc>(std::bind(&PickerNode<TDataType>::changeMultiSelectionType, this));

		this->varMultiSelectionType()->attach(callback3);
	}

	template<typename TDataType>
	PickerNode<TDataType>::~PickerNode()
	{
	}

	template<typename TDataType>
	std::string PickerNode<TDataType>::getNodeType()
	{
		return "Interaction";
	}

	template<typename TDataType>
	void PickerNode<TDataType>::resetStates()
	{
		this->inTopology()->getDataPtr()->updateEdges();

		this->surfaceInteractor->outTriangleIndex()->allocate();
		this->edgeInteractor->outEdgeIndex()->allocate();
		this->pointInteractor->outPointIndex()->allocate();

		this->surfaceInteractor->outOtherTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->surfaceInteractor->outSelectedTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->surfaceInteractor->outSelectedTriangleSet()->getDataPtr()->getTriangles().resize(0);
		this->surfaceInteractor->outOtherTriangleSet()->getDataPtr()->copyFrom(this->inTopology()->getData());

		this->edgeInteractor->outOtherEdgeSet()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
		this->edgeInteractor->outSelectedEdgeSet()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
		this->edgeInteractor->outOtherEdgeSet()->getDataPtr()->copyFrom(this->inTopology()->getData());

		this->pointInteractor->outOtherPointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->pointInteractor->outSelectedPointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->pointInteractor->outOtherPointSet()->getDataPtr()->copyFrom(this->inTopology()->getData());
	}

	template<typename TDataType>
	void PickerNode<TDataType>::changePickingElementType()
	{
		if (this->varPickingElementType()->getValue() == PickingElementTypeSelection::Surface)
		{
			this->surfaceInteractor->varTogglePicker()->setValue(true);
			this->edgeInteractor->varTogglePicker()->setValue(false);
			this->pointInteractor->varTogglePicker()->setValue(false);
		}
		else if (this->varPickingElementType()->getValue() == PickingElementTypeSelection::Edge)
		{
			this->surfaceInteractor->varTogglePicker()->setValue(false);
			this->edgeInteractor->varTogglePicker()->setValue(true);
			this->pointInteractor->varTogglePicker()->setValue(false);
		}
		else if (this->varPickingElementType()->getValue() == PickingElementTypeSelection::Point)
		{
			this->surfaceInteractor->varTogglePicker()->setValue(false);
			this->edgeInteractor->varTogglePicker()->setValue(false);
			this->pointInteractor->varTogglePicker()->setValue(true);
		}
		else if (this->varPickingElementType()->getValue() == PickingElementTypeSelection::All)
		{
			this->surfaceInteractor->varTogglePicker()->setValue(true);
			this->edgeInteractor->varTogglePicker()->setValue(true);
			this->pointInteractor->varTogglePicker()->setValue(true);
		}
		resetStates();
	}

	template<typename TDataType>
	void PickerNode<TDataType>::changePickingType()
	{
		if (this->varPickingType()->getValue() == PickingTypeSelection::Click)
		{
			this->surfaceInteractor->varSurfacePickingType()-> getDataPtr()->setCurrentKey(0);
			this->edgeInteractor->varEdgePickingType()->getDataPtr()->setCurrentKey(0);
			this->pointInteractor->varPointPickingType()->getDataPtr()->setCurrentKey(0);
		}
		else if (this->varPickingType()->getValue() == PickingTypeSelection::Drag)
		{
			this->surfaceInteractor->varSurfacePickingType()->getDataPtr()->setCurrentKey(1);
			this->edgeInteractor->varEdgePickingType()->getDataPtr()->setCurrentKey(1);
			this->pointInteractor->varPointPickingType()->getDataPtr()->setCurrentKey(1);
		}
		else if (this->varPickingType()->getValue() == PickingTypeSelection::Both)
		{
			this->surfaceInteractor->varSurfacePickingType()->getDataPtr()->setCurrentKey(2);
			this->edgeInteractor->varEdgePickingType()->getDataPtr()->setCurrentKey(2);
			this->pointInteractor->varPointPickingType()->getDataPtr()->setCurrentKey(2);
		}
		resetStates();
	}

	template<typename TDataType>
	void PickerNode<TDataType>::changeMultiSelectionType()
	{
		if (this->varMultiSelectionType()->getValue() == MultiSelectionType::OR)
		{
			this->surfaceInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(0);
			this->edgeInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(0);
			this->pointInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(0);
		}
		else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::XOR)
		{
			this->surfaceInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(1);
			this->edgeInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(1);
			this->pointInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(1);
		}
		else if (this->varMultiSelectionType()->getValue() == MultiSelectionType::C)
		{
			this->surfaceInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(2);
			this->edgeInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(2);
			this->pointInteractor->varMultiSelectionType()->getDataPtr()->setCurrentKey(2);
		}
	}

	DEFINE_CLASS(PickerNode);
}