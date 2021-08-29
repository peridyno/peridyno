namespace dyno {

	template<typename T>
	void Array2D<T, DeviceType::GPU>::resize(uint nx, uint ny)
	{
		if (nullptr != m_data) clear();

		cuSafeCall(cudaMallocPitch((void**)&m_data, (size_t*)&m_pitch, (size_t)sizeof(T) * nx, (size_t)ny));
		
		m_nx = nx;	
		m_ny = ny;
	}

	template<typename T>
	void Array2D<T, DeviceType::GPU>::reset()
	{
		cuSafeCall(cudaMemset((void*)m_data, 0, m_pitch * m_ny));
	}

	template<typename T>
	void Array2D<T, DeviceType::GPU>::clear()
	{
		if (m_data != nullptr)
			cuSafeCall(cudaFree((void*)m_data));

		m_nx = 0;
		m_ny = 0;
		m_pitch = 0;
		m_data = nullptr;
	}

	template<typename T>
	void Array2D<T, DeviceType::GPU>::assign(const Array2D<T, DeviceType::GPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size()){
			this->resize(src.nx(), src.ny());
		}

		cuSafeCall(cudaMemcpy2D(m_data, m_pitch, src.begin(), src.pitch(), sizeof(T) * src.nx(), src.ny(), cudaMemcpyDeviceToDevice));
	}

	template<typename T>
	void Array2D<T, DeviceType::GPU>::assign(const Array2D<T, DeviceType::CPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size()) {
			this->resize(src.nx(), src.ny());
		}

		cuSafeCall(cudaMemcpy2D(m_data, m_pitch, src.begin(), sizeof(T) *src.nx(), sizeof(T) * src.nx(), src.ny(), cudaMemcpyHostToDevice));
	}


	template<typename T>
	void Array2D<T, DeviceType::CPU>::resize(uint nx, uint ny)
	{
		if (m_data.size() != 0) clear();
		
		m_data.resize(nx*ny);
		m_nx = nx;
		m_ny = ny;
	}

	template<typename T>
	void Array2D<T, DeviceType::CPU>::reset()
	{
		std::fill(m_data.begin(), m_data.end(), 0);
	}

	template<typename T>
	void dyno::Array2D<T, DeviceType::CPU>::clear()
	{
		m_data.clear();

		m_nx = 0;
		m_ny = 0;
	}

	template<typename T>
	void Array2D<T, DeviceType::CPU>::assign(const Array2D<T, DeviceType::GPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size()) {
			this->resize(src.nx(), src.ny());
		}

		cuSafeCall(cudaMemcpy2D(m_data.data(), sizeof(T) * m_nx, src.begin(), src.pitch(), sizeof(T) *src.nx(), src.ny(), cudaMemcpyDeviceToHost));
	}

	template<typename T>
	void Array2D<T, DeviceType::CPU>::assign(const Array2D<T, DeviceType::CPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size()) {
			this->resize(src.nx(), src.ny());
		}

		cuSafeCall(cudaMemcpy2D(m_data.data(), sizeof(T) *m_nx, src.begin(), sizeof(T) *src.nx(), sizeof(T) *src.nx(), src.ny(), cudaMemcpyHostToHost));
	}
}
