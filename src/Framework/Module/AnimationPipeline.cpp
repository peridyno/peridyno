#include "AnimationPipeline.h"

#include "Node.h"
#include "SceneGraph.h"

namespace dyno
{
	AnimationPipeline::AnimationPipeline(Node* node)
		: Pipeline(node)
	{
	}

	AnimationPipeline::~AnimationPipeline()
	{
	}

	bool AnimationPipeline::printDebugInfo()
	{
		Node* node = this->getParentNode();
		if (node == nullptr || node->getSceneGraph() == nullptr)
			return true;

		return node->getSceneGraph()->isSimulationInfoPrintable();
	}

}