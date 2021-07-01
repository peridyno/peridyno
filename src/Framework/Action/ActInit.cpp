#include "ActInit.h"
#include "Framework/Module.h"
#include "Framework/NumericalModel.h"

namespace dyno
{
	InitAct::InitAct()
	{

	}

	InitAct::~InitAct()
	{

	}

	void InitAct::process(Node* node)
	{
		node->resetStatus();
		node->initialize();

		auto& list = node->getModuleList();
		std::list<std::shared_ptr<Module>>::iterator iter = list.begin();
		for (; iter != list.end(); iter++)
		{
			(*iter)->initialize();
		}

		node->graphicsPipeline()->update();
	}

}