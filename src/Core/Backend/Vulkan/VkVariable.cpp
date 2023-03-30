#include "VkVariable.h"
#include "VkSystem.h"

namespace dyno {

	VkVariable::VkVariable()
	{
	    buffer = std::make_shared<vks::Buffer>();
		ctx = VkSystem::instance()->currentContext();
	}

	VkVariable::~VkVariable()
	{
		// TODO: sovle other issue while destroy buffer here.
		// buffer.destroy();
	}

	VkDescriptorType VkVariable::descriptorType(const VariableType varType)
	{
		switch (varType)
		{
		case DeviceBuffer:
			return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		case Uniform:
			return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		default:
			break;
		}

		return VK_DESCRIPTOR_TYPE_MAX_ENUM;
	}

}