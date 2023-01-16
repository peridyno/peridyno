#pragma once
#include "VkVariable.h"
#include "VulkanBuffer.h"

namespace dyno
{
	template<typename T>
	class VkDeviceArray : public VkVariable
	{
	public:
		VkDeviceArray() {};

		VkDeviceArray(uint32_t num);

		~VkDeviceArray();

		VkResizeType resize(uint32_t num, VkBufferUsageFlags usageFlags = 0);
		inline uint32_t size() const { return m_num; }

		VariableType type() override;

		uint32_t bufferSize() override { return buffer->size; }

		bool bufferUpdated() { return mBufferUpdated; }

		void clear();

		void reset();

	private:
		bool mBufferUpdated = false;
		uint32_t m_num = 0;
	};
}

#include "VkDeviceArray.inl"