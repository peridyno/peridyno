#include "GraphicsPipeline.h"

#include "Node.h"
#include "SceneGraph.h"

namespace dyno
{
	GraphicsPipeline::GraphicsPipeline(Node* node)
		: Pipeline(node)
	{
	}

	GraphicsPipeline::~GraphicsPipeline()
	{
	}

	bool GraphicsPipeline::printDebugInfo()
	{
		Node* node = this->getParentNode();
		if (node == nullptr || node->getSceneGraph() == nullptr)
			return true;

		return node->getSceneGraph()->isRenderingInfoPrintable();
	}

}

