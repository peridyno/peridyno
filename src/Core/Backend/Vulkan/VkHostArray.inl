namespace dyno {

	template<typename T>
	VkHostArray<T>::~VkHostArray()
	{
	}

	template<typename T>
	void VkHostArray<T>::resize(uint32_t num, const T* data)
	{
		uint32_t newSize = num * sizeof(T);
		uint32_t bufferSize = this->bufferSize();

		if (newSize > bufferSize)
		{
			m_num = num;

			buffer->destroy();

			if (num > 0) {
				if (ctx->useMemoryPool) {
					buffer->size = newSize;
					buffer->usageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
					buffer->memoryPropertyFlags =
						VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
					ctx->createBuffer(VkContext::HostPool, buffer, data);
				}
				else {
					ctx->createBuffer(
						VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
						VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
						buffer,
						newSize,
						data);
				}
			}
		}
		else
		{
			m_num = num;
		}
	}

	template<typename T>
	VariableType VkHostArray<T>::type()
	{
		return VariableType::HostBuffer;
	}

	template<typename T>
	void VkHostArray<T>::clear()
	{
		buffer->destroy();
	}

	template<typename T>
	void* VkHostArray<T>::mapped()
	{
		VK_CHECK_RESULT(buffer->map());
		return buffer->mapped;
	}

	template<typename T>
	void VkHostArray<T>::unmap()
	{
		buffer->unmap();
	}
}