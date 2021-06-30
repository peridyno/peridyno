#include "ActAnimate.h"
#include "Framework/Node.h"
#include "Framework/Module.h"
#include "Framework/NumericalModel.h"
#include "Framework/PipelineAnimation.h"
#include "Framework/CollisionModel.h"
#include "Framework/TopologyMapping.h"
#include "Framework/ModuleCustom.h"

namespace dyno
{
	AnimateAct::AnimateAct(float dt)
	{
		m_dt = dt;
	}

	AnimateAct::~AnimateAct()
	{

	}

	void AnimateAct::process(Node* node)
	{
		if (node == NULL)
		{
			Log::sendMessage(Log::Error, "Node is invalid!");
			return;
		}
		if (node->isActive())
		{
			node->updateStatus();

			auto customModules = node->getCustomModuleList();
			for (std::list<std::shared_ptr<CustomModule>>::iterator iter = customModules.begin(); iter != customModules.end(); iter++)
			{
				(*iter)->update();
			}

			node->advance(node->getDt());
			node->updateTopology();

			auto topoModules = node->getTopologyMappingList();
			for (std::list<std::shared_ptr<TopologyMapping>>::iterator iter = topoModules.begin(); iter != topoModules.end(); iter++)
			{
				(*iter)->update();
			}

			/*if (node->getAnimationController() != nullptr)
			{
				node->getAnimationController()->execute();
			}
			else
			{
				auto nModel = node->getNumericalModel();
				if (nModel == NULL)
				{
					Log::sendMessage(Log::Warning, node->getName() + ": No numerical model is set!");
				}
				else
				{
					nModel->step(node->getDt());
					nModel->updateTopology();
				}
				auto cModels = node->getCollisionModelList();
				for (std::list<std::shared_ptr<CollisionModel>>::iterator iter = cModels.begin(); iter != cModels.end(); iter++)
				{
					(*iter)->doCollision();
				}
			}*/
			
		}

// 		if (node->getAnimationController())
// 		{
// 			node->getAnimationController()->execute();
// 		}
	}

}