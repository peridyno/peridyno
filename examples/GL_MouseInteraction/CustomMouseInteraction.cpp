#include "CustomMouseInteraction.h"

namespace dyno
{
	void CustomMouseIteraction::onEvent(PMouseEvent event)
	{
		if (event.actionType == AT_PRESS)
		{
			printf("Mouse pressed \n");
		}
		else if (event.actionType == AT_RELEASE)
		{
			printf("Mouse released \n");
		}
		else
			printf("State repeated \n");
	}
}