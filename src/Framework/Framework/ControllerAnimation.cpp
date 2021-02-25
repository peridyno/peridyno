#include "ControllerAnimation.h"
#include "Node.h"
#include "NumericalModel.h"

namespace dyno
{

IMPLEMENT_CLASS(AnimationController)

AnimationController::AnimationController()
{
}

AnimationController::~AnimationController()
{
}

bool AnimationController::execute()
{
	Node* parent = getParent();
	if (parent == NULL)
	{
		Log::sendMessage(Log::Error, "Parent node is not set!");
		return false;
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

	return true;
}

}