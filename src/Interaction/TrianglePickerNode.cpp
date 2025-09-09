#include "TrianglePickerNode.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"

namespace dyno
{
	IMPLEMENT_TCLASS(TrianglePickerNode, TDataType)

	template<typename TDataType>
	TrianglePickerNode<TDataType>::TrianglePickerNode()
		:Node()
	{
		auto surfaceInteractor = std::make_shared<SurfaceInteraction<TDataType>>();
		auto edgeInteractor = std::make_shared<EdgeInteraction<TDataType>>();
		auto pointInteractor = std::make_shared<PointInteraction<TDataType>>();

		this->inTopology()->connect(surfaceInteractor->inInitialTriangleSet());
		this->inTopology()->connect(edgeInteractor->inInitialEdgeSet());
		this->inTopology()->connect(pointInteractor->inInitialPointSet());

		this->varFloodAngle()->connect(surfaceInteractor->varFloodAngle());
		this->varToggleFlood()->connect(surfaceInteractor->varToggleFlood());
		this->varToggleVisibleFilter()->connect(surfaceInteractor->varToggleVisibleFilter());

		this->varToggleIndexOutput()->connect(surfaceInteractor->varToggleIndexOutput());
		this->varToggleIndexOutput()->connect(edgeInteractor->varToggleIndexOutput());
		this->varToggleIndexOutput()->connect(pointInteractor->varToggleIndexOutput());

		this->varInteractionRadius()->connect(edgeInteractor->varInteractionRadius());
		this->varInteractionRadius()->connect(pointInteractor->varInteractionRadius());

		this->stateTriQuadIndex()->connect(surfaceInteractor->outTriangleIndex());
		this->stateEdgeIndex()->connect(edgeInteractor->outEdgeIndex());
		this->statePointIndex()->connect(pointInteractor->outPointIndex());

		this->stateSur2PointIndex()->connect(surfaceInteractor->outSur2PointIndex());

		this->surfaceInteractor = surfaceInteractor;
		this->edgeInteractor = edgeInteractor;
		this->pointInteractor = pointInteractor;

		this->graphicsPipeline()->pushModule(surfaceInteractor);
		this->graphicsPipeline()->pushModule(edgeInteractor);
		this->graphicsPipeline()->pushModule(pointInteractor);

		auto surfaceRender1 = std::make_shared<GLSurfaceVisualModule>();
		surfaceRender1->setColor(Color(0.2f, 0.48f, 0.75f));
		surfaceRender1->varAlpha()->setValue(0.95f);
		this->surfaceInteractor->outSelectedTriangleSet()->connect(surfaceRender1->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfaceRender1);

		auto surfaceRender2 = std::make_shared<GLSurfaceVisualModule>();
		surfaceRender2->setColor(Color(0.8f, 0.52f, 0.25f));
		surfaceRender2->varAlpha()->setValue(0.85f);
		this->surfaceInteractor->outOtherTriangleSet()->connect(surfaceRender2->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfaceRender2);

		auto edgeRender1 = std::make_shared<GLWireframeVisualModule>();
		this->varEdgeSelectedSize()->connect(edgeRender1->varRadius());
		edgeRender1->setColor(Color(0.8f, 0.0f, 0.0f));
		this->edgeInteractor->outSelectedEdgeSet()->connect(edgeRender1->inEdgeSet());
		this->graphicsPipeline()->pushModule(edgeRender1);

		auto edgeRender2 = std::make_shared<GLWireframeVisualModule>();
		this->varEdgeOtherSize()->connect(edgeRender2->varRadius());
		edgeRender2->setColor(Color(0.0f));
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

		this->varInteractionRadius()->setRange(0.001f , 1.0f);
		this->varInteractionRadius()->setValue(0.01f);
		this->varPointSelectedSize()->setRange(0.0f, 0.5f);
		this->varPointOtherSize()->setRange(0.0f,0.5f);
		this->varEdgeSelectedSize()->setRange(0.0f, 0.5f);
		this->varEdgeOtherSize()->setRange(0.0f, 0.5f);
		this->varFloodAngle()->setRange(0.0f, 180.0f);

		auto callback1 = std::make_shared<FCallBackFunc>(std::bind(&TrianglePickerNode<TDataType>::changePickingElementType, this));

		this->varPickingElementType()->attach(callback1);

		auto callback2 = std::make_shared<FCallBackFunc>(std::bind(&TrianglePickerNode<TDataType>::changePickingType, this));

		this->varPickingType()->attach(callback2);

		auto callback3 = std::make_shared<FCallBackFunc>(std::bind(&TrianglePickerNode<TDataType>::changeMultiSelectionType, this));

		this->varMultiSelectionType()->attach(callback3);

		this->surfaceInteractor->outTriangleIndex()->allocate();
		this->edgeInteractor->outEdgeIndex()->allocate();
		this->pointInteractor->outPointIndex()->allocate();
		this->surfaceInteractor->outSur2PointIndex()->allocate();
	}

	template<typename TDataType>
	TrianglePickerNode<TDataType>::~TrianglePickerNode()
	{
	}

	template<typename TDataType>
	std::string TrianglePickerNode<TDataType>::getNodeType()
	{
		return "Interaction";
	}

	template<typename TDataType>
	void TrianglePickerNode<TDataType>::resetStates()
	{
		this->inTopology()->getDataPtr()->update();

		this->surfaceInteractor->outTriangleIndex()->allocate();
		this->edgeInteractor->outEdgeIndex()->allocate();
		this->pointInteractor->outPointIndex()->allocate();

		this->surfaceInteractor->outOtherTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->surfaceInteractor->outSelectedTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->surfaceInteractor->outSelectedTriangleSet()->getDataPtr()->triangleIndices().resize(0);
		this->surfaceInteractor->outOtherTriangleSet()->getDataPtr()->copyFrom(this->inTopology()->getData());

		this->edgeInteractor->outOtherEdgeSet()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
		this->edgeInteractor->outSelectedEdgeSet()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
		this->edgeInteractor->outOtherEdgeSet()->getDataPtr()->copyFrom(this->inTopology()->getData());

		this->pointInteractor->outOtherPointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->pointInteractor->outSelectedPointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->pointInteractor->outOtherPointSet()->getDataPtr()->copyFrom(this->inTopology()->getData());
	}

	template<typename TDataType>
	void TrianglePickerNode<TDataType>::changePickingElementType()
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
	void TrianglePickerNode<TDataType>::changePickingType()
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
	void TrianglePickerNode<TDataType>::changeMultiSelectionType()
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

	DEFINE_CLASS(TrianglePickerNode);
}