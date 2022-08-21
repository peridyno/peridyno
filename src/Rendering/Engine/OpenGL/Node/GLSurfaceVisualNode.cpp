#include "GLSurfaceVisualNode.h"

#include <GLSurfaceVisualModule.h>

namespace dyno
{
	template<typename TDataType>
	GLSurfaceVisualNode<TDataType>::GLSurfaceVisualNode()
		: Node()
	{
		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		this->varColor()->connect(sRender->varBaseColor());
		this->inTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);
	}

	template<typename TDataType>
	GLSurfaceVisualNode<TDataType>::~GLSurfaceVisualNode()
	{
		printf("GLSurfaceVisualNode released \n");
	}

	template<typename TDataType>
	std::string dyno::GLSurfaceVisualNode<TDataType>::caption()
	{
		return "Surface Visualizer";
	}

	template<typename TDataType>
	std::string GLSurfaceVisualNode<TDataType>::getNodeType()
	{
		return "Visualization";
	}

	template<typename TDataType>
	void GLSurfaceVisualNode<TDataType>::resetStates()
	{
		this->update();
	}

	DEFINE_CLASS(GLSurfaceVisualNode);
}
