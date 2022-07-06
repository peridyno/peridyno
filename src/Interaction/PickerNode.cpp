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
		auto mouseInteractor = std::make_shared<PickerInteraction<TDataType>>();
		this->stateInTopology()->connect(mouseInteractor->inInitialTriangleSet());

		this->stateSelectedTriangleSet()->connect(mouseInteractor->outSelectedTriangleSet());
		this->stateOtherTriangleSet()->connect(mouseInteractor->outOtherTriangleSet());
		this->stateSelectedEdgeSet()->connect(mouseInteractor->outSelectedEdgeSet());
		this->stateOtherEdgeSet()->connect(mouseInteractor->outOtherEdgeSet());
		this->stateSelectedPointSet()->connect(mouseInteractor->outSelectedPointSet());
		this->stateOtherPointSet()->connect(mouseInteractor->outOtherPointSet());
		this->stateTriangleIndex()->connect(mouseInteractor->outTriangleIndex());
		this->stateEdgeIndex()->connect(mouseInteractor->outEdgeIndex());
		this->statePointIndex()->connect(mouseInteractor->outPointIndex());

		this->varInterationRadius()->connect(mouseInteractor->varInterationRadius());
		this->varToggleSurfacePicker()->connect(mouseInteractor->varToggleSurfacePicker());
		this->varToggleEdgePicker()->connect(mouseInteractor->varToggleEdgePicker());
		this->varTogglePointPicker()->connect(mouseInteractor->varTogglePointPicker());

		this->varToggleMultiSelect()->connect(mouseInteractor->varToggleMultiSelect());

		this->mouseInteractor = mouseInteractor;

		this->stateInTopology()->promoteInput();

		this->graphicsPipeline()->pushModule(mouseInteractor);

		auto surfaceRender1 = std::make_shared<GLSurfaceVisualModule>();
		this->varSelectedTriangleColor()->connect(surfaceRender1->varBaseColor());
		this->stateSelectedTriangleSet()->connect(surfaceRender1->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfaceRender1);

		auto surfaceRender2 = std::make_shared<GLSurfaceVisualModule>();
		this->varOtherTriangleColor()->connect(surfaceRender2->varBaseColor());
		this->stateOtherTriangleSet()->connect(surfaceRender2->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfaceRender2);

		auto edgeRender1 = std::make_shared<GLWireframeVisualModule>();
		this->varSelectedEdgeColor()->connect(edgeRender1->varBaseColor());
		this->stateSelectedEdgeSet()->connect(edgeRender1->inEdgeSet());
		this->graphicsPipeline()->pushModule(edgeRender1);

		auto edgeRender2 = std::make_shared<GLWireframeVisualModule>();
		this->varOtherEdgeColor()->connect(edgeRender2->varBaseColor());
		this->stateOtherEdgeSet()->connect(edgeRender2->inEdgeSet());
		this->graphicsPipeline()->pushModule(edgeRender2);

		auto pointRender1 = std::make_shared<GLPointVisualModule>();
		this->varSelectedSize()->connect(pointRender1->varPointSize());
		this->varSelectedPointColor()->connect(pointRender1->varBaseColor());
		this->stateSelectedPointSet()->connect(pointRender1->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender1);

		auto pointRender2 = std::make_shared<GLPointVisualModule>();
		this->varOtherSize()->connect(pointRender2->varPointSize());
		this->varOtherPointColor()->connect(pointRender2->varBaseColor());
		this->stateOtherPointSet()->connect(pointRender2->inPointSet());
		this->graphicsPipeline()->pushModule(pointRender2);
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
		this->stateInTopology()->getDataPtr()->updateEdges();

		this->stateTriangleIndex()->allocate();
		this->stateEdgeIndex()->allocate();
		this->statePointIndex()->allocate();

		this->stateOtherTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->stateSelectedTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->stateSelectedTriangleSet()->getDataPtr()->getTriangles().resize(0);
		this->stateOtherTriangleSet()->getDataPtr()->copyFrom(this->stateInTopology()->getData());

		this->stateOtherEdgeSet()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
		this->stateSelectedEdgeSet()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());
		this->stateOtherEdgeSet()->getDataPtr()->copyFrom(this->stateInTopology()->getData());

		this->stateOtherPointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->stateSelectedPointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->stateOtherPointSet()->getDataPtr()->copyFrom(this->stateInTopology()->getData());
	}

	DEFINE_CLASS(PickerNode);
}