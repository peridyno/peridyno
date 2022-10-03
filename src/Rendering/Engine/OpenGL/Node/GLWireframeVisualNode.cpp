#include "GLWireframeVisualNode.h"

#include <GLWireframeVisualModule.h>

namespace dyno
{
	template<typename TDataType>
	GLWireframeVisualNode<TDataType>::GLWireframeVisualNode()
		: Node()
	{
		auto wRender = std::make_shared<GLWireframeVisualModule>();
		this->varColor()->connect(wRender->varBaseColor());
		this->inTriangleSet()->connect(wRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(wRender);
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
