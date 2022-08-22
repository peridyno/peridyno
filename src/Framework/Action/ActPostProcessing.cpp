#include "ActPostProcessing.h"

namespace dyno
{
	
	PostProcessing::PostProcessing()
	{

	}

	PostProcessing::~PostProcessing()
	{

	}

	void PostProcessing::process(Node* node)
	{
		auto mList = node->getModuleList();
		int cnt = 0;
		for (auto iter = mList.begin(); iter != mList.end(); iter++)
		{
			//printf("iter %s ::: %s\n", (*iter)->getName(),(*iter)->getModuleType());
			if (std::string("OutputModule").compare((*iter)->getModuleType()) == 0)
			{
				
				(*iter)->update();
			}
		}
	}
}