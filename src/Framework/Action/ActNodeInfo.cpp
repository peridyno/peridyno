#include "ActNodeInfo.h"

namespace dyno
{
	
	NodeInfoAct::NodeInfoAct()
	{

	}

	NodeInfoAct::~NodeInfoAct()
	{

	}

	void NodeInfoAct::process(Node* node)
	{
		std::cout << node->getName() << std::endl;
	}

}