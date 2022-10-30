#include "VkUniform.h"

namespace px {

	template<typename T>
	VkUniform<T>::VkUniform()
		: VkVariable()
	{
		if (ctx->useMemoryPool) {
			buffer->size = sizeof(T);
			buffer->usageFlags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
			buffer->memoryPropertyFlags =
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			ctx->createBuffer(VkContext::UniformPool, buffer);
		} else {
			ctx->createBuffer(
					VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					buffer,
					sizeof(T));
		}
		VK_CHECK_RESULT(buffer->map());
	}

	template<typename T>
	VkUniform<T>::~VkUniform()
	{
		buffer->destroy();
	}

	template<typename T>
	void VkUniform<T>::setValue(T val)
	{
		memcpy(buffer->mapped, &val, sizeof(T));
	}

	template<typename T>
	VariableType VkUniform<T>::type()
	{
		return VariableType::Uniform;
	}

}