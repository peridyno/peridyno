#include "GLWireframeVisualNode.h"

#include <GLWireframeVisualModule.h>

namespace dyno
{
	template<typename TDataType>
	GLWireframeVisualNode<TDataType>::GLWireframeVisualNode()
		: Node()
	{
		auto sRender = std::make_shared<GLWireframeVisualModule>();
		sRender->setColor(Vec3f(0, 0, 0));
		this->inTriangleSet()->connect(sRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(sRender);
	}

	template<typename TDataType>
	GLWireframeVisualNode<TDataType>::~GLWireframeVisualNode()
	{
		printf("GLWireframeVisualNode released \n");
	}

	template<typename TDataType>
	std::string GLWireframeVisualNode<TDataType>::getNodeType()
	{
		return "Visualization";
	}

	template<typename TDataType>
	void GLWireframeVisualNode<TDataType>::resetStates()
	{
		this->update();
	}

	DEFINE_CLASS(GLWireframeVisualNode);
}
