#pragma once
#include "VkVariable.h"
#include "VulkanBuffer.h"

namespace dyno {

	template<typename T>
	class VkUniform : public VkVariable
	{
	public:
		VkUniform();
		~VkUniform();

		void setValue(T val);

		VariableType type() override;

		uint32_t bufferSize() override { return sizeof(T); }

	protected:
	};
}

#include "VkUniform.inl"