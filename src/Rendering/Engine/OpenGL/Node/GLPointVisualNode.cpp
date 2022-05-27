#include "GLPointVisualNode.h"

#include <Module/CalculateNorm.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>

namespace dyno
{
	template<typename TDataType>
	GLPointVisualNode<TDataType>::GLPointVisualNode()
		: Node()
	{
		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMax()->setValue(5.0f);

		this->inVector()->connect(calculateNorm->inVec());
		calculateNorm->outNorm()->connect(colorMapper->inScalar());

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Vec3f(1, 0, 0));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
		ptRender->setColorMapRange(0, 5);

		this->inPoints()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());
		
		this->graphicsPipeline()->pushModule(calculateNorm);
		this->graphicsPipeline()->pushModule(colorMapper);
		this->graphicsPipeline()->pushModule(ptRender);
	}

	template<typename TDataType>
	GLPointVisualNode<TDataType>::~GLPointVisualNode()
	{
		printf("GLPointVisualNode released \n");
	}

	template<typename TDataType>
	std::string GLPointVisualNode<TDataType>::getNodeType()
	{
		return "Visualization";
	}

	template<typename TDataType>
	void GLPointVisualNode<TDataType>::resetStates()
	{
		this->animationPipeline()->update();
	}

	DEFINE_CLASS(GLPointVisualNode);
}
