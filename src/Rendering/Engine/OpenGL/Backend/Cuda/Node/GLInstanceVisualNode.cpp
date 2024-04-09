#include "GLInstanceVisualNode.h"

#include <GLInstancePhotorealisticRender.h>

namespace dyno
{
	template<typename TDataType>
	GLInstanceVisualNode<TDataType>::GLInstanceVisualNode()
		: Node()
	{
		auto sRender = std::make_shared<GLInstancePhotorealisticRender>();
		this->inInstances()->connect(sRender->inInstances());
		this->inTextureMesh()->connect(sRender->inTextureMesh());

		this->graphicsPipeline()->pushModule(sRender);
		this->setAutoSync(true);
	}

	template<typename TDataType>
	GLInstanceVisualNode<TDataType>::~GLInstanceVisualNode()
	{
		printf("GLInstanceVisualNode released \n");
	}

	template<typename TDataType>
	std::string dyno::GLInstanceVisualNode<TDataType>::caption()
	{
		return "Instance Visualizer";
	}

	template<typename TDataType>
	std::string GLInstanceVisualNode<TDataType>::getNodeType()
	{
		return "Visualization";
	}

	template<typename TDataType>
	void GLInstanceVisualNode<TDataType>::resetStates()
	{
		this->update();
	}

	DEFINE_CLASS(GLInstanceVisualNode);
}
