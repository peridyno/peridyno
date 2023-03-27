#pragma once
#include "VkVariable.h"
#include "VulkanBuffer.h"

namespace dyno {

	template<typename T>
	class VkHostArray : public VkVariable
	{

	public:
		VkHostArray() {};

		~VkHostArray();

		void resize(uint32_t num, const T* data = nullptr);
		inline uint32_t size() const { return m_num; }

		VariableType type() override;

		uint32_t bufferSize() override { return sizeof(T)*m_num; }

		void clear();

		void* mapped();
		void unmap();

	private:
		uint32_t m_num = 0;
	};
}

#include "VkHostArray.inl"