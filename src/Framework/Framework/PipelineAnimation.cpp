#include "Node.h"
#include "PipelineAnimation.h"
#include "NumericalModel.h"

namespace dyno
{
	AnimationPipeline::AnimationPipeline(Node* node)
		: Pipeline(node)
	{
	}

	AnimationPipeline::~AnimationPipeline()
	{
	}

	void AnimationPipeline::updateImpl()
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Parent node is not set!");
			return;
		}
		if (parent->isActive())
		{
			auto nModel = parent->getNumericalModel();
			if (nModel == NULL)
			{
				Log::sendMessage(Log::Warning, parent->getName() + ": No numerical model is set!");
			}
			else
			{
				nModel->step(parent->getDt());
				nModel->updateTopology();
			}
		}
	}
}