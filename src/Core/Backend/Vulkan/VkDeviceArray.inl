namespace dyno {

	template<typename T>
	VkDeviceArray<T>::~VkDeviceArray()
	{
	}

	template<typename T>
	VkDeviceArray<T>::VkDeviceArray(uint32_t num)
	{
		this->resize(num);
	}

	template<typename T>
	VkResizeType VkDeviceArray<T>::resize(uint32_t num, VkBufferUsageFlags usageFlags)
	{
		uint32_t newSize = num * sizeof(T);
		uint32_t bufferSize = this->bufferSize();

		if (newSize > bufferSize)
		{
			m_num = num;

			buffer = std::make_shared<vks::Buffer>();

			if (num > 0)
			{
				if (ctx->useMemPool()) {
					buffer->size = newSize;
					buffer->usageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
						VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | usageFlags;
					buffer->memoryPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
					ctx->createBuffer(VkContext::DevicePool, buffer);
				}
				else {
					ctx->createBuffer(
						usageFlags | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
						VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						buffer,
						newSize);
				}
				vkFill(*this, 0);
				VkCompContext::current().registerBuffer(buffer);
			}

			return VK_BUFFER_REALLOCATED;
		}
		else
		{
			m_num = num;
			return VK_BUFFER_REUSED;
		}
	}

	template<typename T>
	VariableType VkDeviceArray<T>::type()
	{
		return VariableType::DeviceBuffer;
	}

	template<typename T>
	void VkDeviceArray<T>::clear()
	{
		m_num = 0;
		// do not call destroy, may be used in command buffer
		buffer = std::make_shared<vks::Buffer>();
	}

	template<typename T>
	void VkDeviceArray<T>::reset()
	{
		m_num = 0;
	}
}