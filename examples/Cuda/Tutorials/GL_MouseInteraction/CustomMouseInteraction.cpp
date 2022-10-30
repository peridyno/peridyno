#include "CustomMouseInteraction.h"

namespace dyno
{
	void CustomMouseInteraction::onEvent(PMouseEvent event)
	{
		if (event.actionType == AT_PRESS)
		{
			printf("Mouse pressed: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
		}
		else if (event.actionType == AT_RELEASE)
		{
			printf("Mouse released: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);

		}
		else
		{
			printf("%f %f \n", event.x, event.y);
			printf("Mouse repeated: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x, event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y, event.ray.direction.z);
		}
	}
}