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


	VkDeviceAddress VkVariable::bufferAddress() const {
		static_assert(sizeof(VkDeviceAddress) == sizeof(uint64_t));
		VkBufferDeviceAddressInfo info {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
		info.buffer = bufferHandle();
		if (info.buffer == VK_NULL_HANDLE)
			return VK_NULL_HANDLE;
		return vkGetBufferDeviceAddress(currentContext()->deviceHandle(), &info);
	}
}