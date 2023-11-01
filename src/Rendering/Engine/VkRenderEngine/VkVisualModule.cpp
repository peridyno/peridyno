#include "VkVisualModule.h"

namespace dyno
{
	VkVisualModule::VkVisualModule()
		: VisualModule()
	{
	}

	VkVisualModule::~VkVisualModule()
	{
	}

	void VkVisualModule::updateImpl()
	{
		this->updateGraphicsContext();
	}

}