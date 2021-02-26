namespace dyno 
{
	template<typename T, DeviceType deviceType>
	void Array<T, deviceType>::resize(const size_t n)
	{
		//		assert(n >= 1);
		if (NULL != m_data) release();
		if (n <= 0)
		{
			m_totalNum = 0;
		}
		else
		{
			m_totalNum = n;
			allocMemory();
		}
	}

	template<typename T, DeviceType deviceType>
	void Array<T, deviceType>::release()
	{
		if (m_data != NULL)
		{
			m_alloc->releaseMemory((void**)&m_data);
		}

		m_data = NULL;
		m_totalNum = 0;
	}

	template<typename T, DeviceType deviceType>
	void Array<T, deviceType>::allocMemory()
	{
		m_alloc->allocMemory1D((void**)&m_data, m_totalNum, sizeof(T));

		reset();
	}

	template<typename T, DeviceType deviceType>
	void Array<T, deviceType>::reset()
	{
		m_alloc->initMemory((void*)m_data, 0, m_totalNum * sizeof(T));
	}
}
