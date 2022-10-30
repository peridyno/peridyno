namespace px {

	template<typename T>
	VkHostArray2D<T>::~VkHostArray2D()
	{
	}

	template<typename T>
	VkHostArray2D<T>::VkHostArray2D(uint32_t nx, uint32_t ny)
	{
		this->resize(nx, ny);
	}

	template<typename T>
	uint32_t VkHostArray2D<T>::index(uint32_t i, uint32_t j)
	{
		return i + j * m_nx;
	}

	template<typename T>
	void VkHostArray2D<T>::resize(uint32_t nx, uint32_t ny, const T* data)
	{
		buffer->destroy();

		m_nx = nx;
		m_ny = ny;
		m_num = nx * ny;

		if (m_num > 0) {
			if (ctx->useMemoryPool) {
				buffer->size = m_num * sizeof(T);
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
					m_num * sizeof(T),
					data);
			}
		}
	}

	template<typename T>
	VariableType VkHostArray2D<T>::type()
	{
		return VariableType::HostBuffer;
	}

	template<typename T>
	void VkHostArray2D<T>::clear()
	{
		m_nx = 0;
		m_ny = 0;
		m_num = 0;
		buffer->destroy();
	}

	template<typename T>
	void* VkHostArray2D<T>::mapped()
	{
		VK_CHECK_RESULT(buffer->map());
		return buffer->mapped;
	}

	template<typename T>
	void VkHostArray2D<T>::unmap()
	{
		buffer->unmap();
	}

	template<typename T>
	HostArray2DInfo VkHostArray2D<T>::getInfo()
	{
		HostArray2DInfo info;
		info.nx = m_nx;
		info.ny = m_ny;
		info.nxy = m_nx * m_ny;

		return info;
	}
}