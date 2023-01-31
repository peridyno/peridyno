#include "GLCommonPointVisualNode.h"

#include <Module/CalculateNorm.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>

namespace dyno
{
	template<typename TDataType>
	GLCommonPointVisualNode<TDataType>::GLCommonPointVisualNode()
		: Node()
	{
		auto pRender = std::make_shared<GLPointVisualModule>();
		this->varColor()->connect(pRender->varBaseColor());
		this->varPointSize()->connect(pRender->varPointSize());
		this->inPointSet()->connect(pRender->inPointSet());
		this->graphicsPipeline()->pushModule(pRender);
	}

	template<typename TDataType>
	GLCommonPointVisualNode<TDataType>::~GLCommonPointVisualNode()
	{
		printf("GLCommonPointVisualNode released \n");
	}

	template<typename TDataType>
	std::string GLCommonPointVisualNode<TDataType>::getNodeType()
	{
		return "Visualization";
	}

	template<typename TDataType>
	void GLCommonPointVisualNode<TDataType>::resetStates()
	{
		this->update();
	}

	DEFINE_CLASS(GLCommonPointVisualNode);
}
