#pragma once
#include "Platform.h"
#include "vulkan/vulkan.h"

#include "VkContext.h"

namespace dyno {

	/*!
	*	\enum	ArgumentType
	*	\brief	Different types of Vukan kernel programs.
	*/
	enum VariableType
	{
		DeviceBuffer,	//!< Device buffer
		HostBuffer,		//!< Host buffer
		Constant,		//!< Constant variable
		Uniform			//!< Uniform variable
	};

	enum VkResizeType
	{
		VK_BUFFER_REALLOCATED = 0x00000000,
		VK_BUFFER_REUSED = 0x00000001,
		VK_FAILED = 0xFFFFFFFF
	};

	class VkVariable {
	public:
		VkVariable();
		~VkVariable();

		VkContext* currentContext() const { return ctx; }

		VkDescriptorBufferInfo& getDescriptor() { return buffer->descriptor; }

		VkBuffer bufferHandle() const { return buffer->buffer; }

		VkDeviceAddress bufferAddress() const;

		virtual VariableType type() = 0;

		static VkDescriptorType descriptorType(const VariableType varType);

		virtual uint32_t bufferSize() = 0;

		virtual void* data() const { return nullptr; }

		std::shared_ptr<vks::Buffer> bufferSp() { return buffer; }

	protected:
		VkContext* ctx = nullptr;

		std::shared_ptr<vks::Buffer> buffer;
	};
}