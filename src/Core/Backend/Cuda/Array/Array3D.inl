namespace dyno {

	template<typename T>
	void Array3D<T, DeviceType::CPU>::assign(const Array3D<T, DeviceType::GPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size() || m_nz != src.size()) {
			this->resize(src.nx(), src.ny(), src.nz());
		}

		cuSafeCall(cudaMemcpy2D(m_data.data(), sizeof(T) * m_nx, src.begin(), src.pitch(), sizeof(T) * src.nx(), src.ny() * src.nz(), cudaMemcpyDeviceToHost));
	}

	template<typename T>
	class Array3D<T, DeviceType::GPU>
	{
	public:
		Array3D()
		{};

		Array3D(uint nx, uint ny, uint nz)
		{
			this->resize(nx, ny, nz);
		};

		/*!
			*	\brief	Should not release data here, call Release() explicitly.
			*/
		~Array3D() { };

		void resize(const uint nx, const uint ny, const uint nz);

		void reset();

		void clear();

		inline T* begin() const { return m_data; }

		DYN_FUNC inline uint nx() const { return m_nx; }
		DYN_FUNC inline uint ny() const { return m_ny; }
		DYN_FUNC inline uint nz() const { return m_nz; }
		DYN_FUNC inline uint pitch() const { return m_pitch_x; }

		DYN_FUNC inline T operator () (const int i, const int j, const int k) const
		{
			char* addr = (char*)m_data;
			addr += (j * m_pitch_x + k * m_nxy);
			return ((T*)addr)[i];
		}

		DYN_FUNC inline T& operator () (const int i, const int j, const int k)
		{
			char* addr = (char*)m_data;
			addr += (j * m_pitch_x + k * m_nxy);
			return ((T*)addr)[i];
		}

		DYN_FUNC inline T operator [] (const int id) const
		{
			return m_data[id];
		}

		DYN_FUNC inline T& operator [] (const int id)
		{
			return m_data[id];
		}

		DYN_FUNC inline size_t index(const uint i, const uint j, const uint k) const
		{
			return i + j * m_nx + k * m_nx * m_ny;
		}

		DYN_FUNC inline size_t size() const { return m_nx * m_ny * m_nz; }
		DYN_FUNC inline bool isCPU() const { return false; }
		DYN_FUNC inline bool isGPU() const { return true; }

		void assign(const Array3D<T, DeviceType::GPU>& src);
		void assign(const Array3D<T, DeviceType::CPU>& src);

	private:
		uint m_nx = 0;
		uint m_pitch_x = 0;

		uint m_ny = 0;
		uint m_nz = 0;
		uint m_nxy = 0;
		T* m_data = nullptr;
	};

	template<typename T>
	using DArray3D = Array3D<T, DeviceType::GPU>;

	typedef DArray3D<float>	Grid1f;
	typedef DArray3D<float3> Grid3f;
	typedef DArray3D<bool> Grid1b;


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
