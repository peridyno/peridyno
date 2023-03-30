namespace dyno {

	template<typename T>
	void Array3D<T, DeviceType::CPU>::resize(uint nx, uint ny, uint nz)
	{
		m_data.clear();
		m_nx = nx;	m_ny = ny;	m_nz = nz; m_nxy = nx * ny;

		m_data.resize((size_t)nx*ny*nz);
	}

	template<typename T>
	void Array3D<T, DeviceType::CPU>::reset()
	{
		std::fill(m_data.begin(), m_data.end(), 0);
	}

	template<typename T>
	void Array3D<T, DeviceType::CPU>::clear()
	{
		m_nx = 0;
		m_ny = 0;
		m_nz = 0;
		m_data.clear();
	}

	template<typename T>
	void Array3D<T, DeviceType::CPU>::assign(uint nx, uint ny, uint nz, const T& val)
	{
		if (m_nx != nx || m_ny != ny || m_nz != nz) {
			this->resize(nx, ny, nz);
		}

		m_data.assign(m_data.size(), val);
	}

	template<typename T>
	void Array3D<T, DeviceType::CPU>::assign(const T& val)
	{
		m_data.assign(m_data.size(), val);
	}

	template<typename T>
	void Array3D<T, DeviceType::CPU>::assign(const Array3D<T, DeviceType::GPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size() || m_nz != src.size()) {
			this->resize(src.nx(), src.ny(), src.nz());
		}

		cuSafeCall(cudaMemcpy2D(m_data.data(), sizeof(T) *m_nx, src.begin(), src.pitch(), sizeof(T) *src.nx(), src.ny()*src.nz(), cudaMemcpyDeviceToHost));
	}

	template<typename T>
	void Array3D<T, DeviceType::CPU>::assign(const Array3D<T, DeviceType::CPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size() || m_nz != src.size()) {
			this->resize(src.nx(), src.ny(), src.nz());
		}

		cuSafeCall(cudaMemcpy2D(m_data.data(), sizeof(T) *m_nx, src.begin(), sizeof(T) *src.nx(), sizeof(T) *src.nx(), src.ny()*src.nz(), cudaMemcpyHostToHost));
	}


	template<typename T>
	void Array3D<T, DeviceType::GPU>::resize(const uint nx, const uint ny, const uint nz)
	{
		if (NULL != m_data) clear();
		
		cuSafeCall(cudaMallocPitch((void**)&m_data, (size_t*)&m_pitch_x, (size_t)sizeof(T) * nx, (size_t)ny*nz));

		//TODO: check whether it has problem when m_pitch_x is not divisible by sizeof(T)
		m_nx = nx;	m_ny = ny;	m_nz = nz;	
		m_nxy = m_pitch_x * m_ny;
	}

	template<typename T>
	void Array3D<T, DeviceType::GPU>::reset()
	{
		cuSafeCall(cudaMemset(m_data, 0, m_nxy * m_nz));
	}

	template<typename T>
	void Array3D<T, DeviceType::GPU>::clear()
	{
		if(m_data != nullptr) cuSafeCall(cudaFree(m_data));

		m_data = nullptr;
		m_nx = 0;
		m_ny = 0;
		m_nz = 0;
		m_nxy = 0;
	}

	template<typename T>
	void Array3D<T, DeviceType::GPU>::assign(const Array3D<T, DeviceType::GPU>& src)
	{
		if (m_nx != src.nx() || m_ny != src.ny() || m_nz != src.nz()) {
			this->resize(src.nx(), src.ny(), src.nz());
		}

		cuSafeCall(cudaMemcpy2D(m_data, m_pitch_x, src.begin(), src.pitch(), sizeof(T) *src.nx(), src.ny()*src.nz(), cudaMemcpyDeviceToDevice));
	}

	template<typename T>
	void Array3D<T, DeviceType::GPU>::assign(const Array3D<T, DeviceType::CPU>& src)
	{
		if (m_nx != src.nx() || m_ny != src.ny() || m_nz != src.nz()) {
			this->resize(src.nx(), src.ny(), src.nz());
		}

		cuSafeCall(cudaMemcpy2D(m_data, m_pitch_x, src.begin(), sizeof(T) *src.nx(), sizeof(T) *src.nx(), src.ny()*src.nz(), cudaMemcpyHostToDevice));
	}
}
