#include "ActReset.h"
#include "Node.h"


namespace dyno
{
	
	ResetAct::ResetAct()
	{

	}

	ResetAct::~ResetAct()
	{

	}

	void ResetAct::process(Node* node)
	{
		if (node == NULL)
		{
			Log::sendMessage(Log::Error, "Node is invalid!");
			return;
		}

		node->reset();
	}

}