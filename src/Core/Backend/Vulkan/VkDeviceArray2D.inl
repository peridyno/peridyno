namespace px {

	template<typename T>
	VkDeviceArray2D<T>::~VkDeviceArray2D()
	{
	}

	template<typename T>
	VkDeviceArray2D<T>::VkDeviceArray2D(uint32_t nx, uint32_t ny)
	{
		this->resize(nx, ny);
	}

	template<typename T>
	void VkDeviceArray2D<T>::resize(uint32_t nx, uint32_t ny, VkBufferUsageFlags usageFlags)
	{
		buffer->destroy();

		m_nx = nx;
		m_ny = ny;

		m_num = nx * ny;

		if (m_num > 0)
		{
			if (ctx->useMemoryPool) {
				buffer->size = m_num * sizeof(T);
				buffer->usageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT | usageFlags;
				buffer->memoryPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
				ctx->createBuffer(VkContext::DevicePool, buffer);
			}
			else {
				ctx->createBuffer(
					usageFlags | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					buffer,
					m_num * sizeof(T));
			}
		}
	}

	template<typename T>
	VariableType VkDeviceArray2D<T>::type()
	{
		return VariableType::DeviceBuffer;
	}

	template<typename T>
	void VkDeviceArray2D<T>::clear()
	{
		m_num = 0;
		m_nx = 0;
		m_ny = 0;
		buffer->destroy();
	}

	template<typename T>
	Array2DInfo VkDeviceArray2D<T>::getInfo()
	{
		Array2DInfo info;
		info.nx = m_nx;
		info.ny = m_ny;

		return info;
	}
}