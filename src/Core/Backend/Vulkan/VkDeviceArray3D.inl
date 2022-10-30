namespace px {

	template<typename T>
	VkDeviceArray3D<T>::~VkDeviceArray3D()
	{
	}

	template<typename T>
	VkDeviceArray3D<T>::VkDeviceArray3D(uint32_t nx, uint32_t ny, uint32_t nz)
	{
		this->resize(nx, ny, nz);
	}

	template<typename T>
	void VkDeviceArray3D<T>::resize(uint32_t nx, uint32_t ny, uint32_t nz, VkBufferUsageFlags usageFlags)
	{
		buffer->destroy();

		m_nx = nx;
		m_ny = ny;
		m_nz = nz;
		m_num = nx * ny * nz;

		if (m_num > 0)
		{
			if (ctx->useMemoryPool) {
				buffer->size = m_num * sizeof(T);
				buffer->usageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
									 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | usageFlags;
				buffer->memoryPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
				ctx->createBuffer(VkContext::DevicePool, buffer);
			} else {
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
	uint32_t VkDeviceArray3D<T>::index(uint32_t i, uint32_t j, uint32_t k)
	{
		return i + j * m_nx + k * m_nx * m_ny;
	}

	template<typename T>
	VariableType VkDeviceArray3D<T>::type()
	{
		return VariableType::DeviceBuffer;
	}

	template<typename T>
	void VkDeviceArray3D<T>::clear()
	{
		m_nx = 0;
		m_ny = 0;
		m_nz = 0;
		m_num = 0;
		buffer->destroy();
	}

	template<typename T>
	Array3DInfo VkDeviceArray3D<T>::getInfo()
	{
		Array3DInfo info;
		info.nx = m_nx;
		info.ny = m_ny;
		info.nz = m_nz;
		info.nxy = m_nx * m_ny;

		return info;
	}
}