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
}