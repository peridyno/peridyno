namespace dyno {

	template<typename T>
	void Array2D<T, DeviceType::CPU>::assign(const Array2D<T, DeviceType::GPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size()) {
			this->resize(src.nx(), src.ny());
		}

		cuSafeCall(cudaMemcpy2D(m_data.data(), sizeof(T) * m_nx, src.begin(), src.pitch(), sizeof(T) * src.nx(), src.ny(), cudaMemcpyDeviceToHost));
	}

	template<typename T>
	class Array2D<T, DeviceType::GPU>
	{
	public:
		Array2D() {};

		Array2D(uint nx, uint ny)
		{
			this->resize(nx, ny);
		};

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		DYN_FUNC ~Array2D() {};

		void resize(uint nx, uint ny);

		void reset();

		void clear();

		inline T* begin() const { return m_data; }

		// Called by silling
		inline uint64 address() const {
			return uint64(m_data);
		}
		DYN_FUNC inline uint nx() const { return m_nx; }
		DYN_FUNC inline uint ny() const { return m_ny; }
		DYN_FUNC inline uint pitch() const { return m_pitch; }

		GPU_FUNC inline T operator () (const uint i, const uint j) const
		{
			char* addr = (char*)m_data;
			addr += j * m_pitch;

			return ((T*)addr)[i];
			//return m_data[i + j* m_pitch];
		}

		GPU_FUNC inline T& operator () (const uint i, const uint j)
		{
			char* addr = (char*)m_data;
			addr += j * m_pitch;

			return ((T*)addr)[i];

			//return m_data[i + j* m_pitch];
		}

		DYN_FUNC inline int index(const uint i, const uint j) const
		{
			return i + j * m_nx;
		}

		GPU_FUNC inline T operator [] (const uint id) const
		{
			return m_data[id];
		}

		GPU_FUNC inline T& operator [] (const uint id)
		{
			return m_data[id];
		}

		DYN_FUNC inline uint size() const { return m_nx * m_ny; }
		DYN_FUNC inline bool isCPU() const { return false; }
		DYN_FUNC inline bool isGPU() const { return true; }

		void assign(const Array2D<T, DeviceType::GPU>& src);
		void assign(const Array2D<T, DeviceType::CPU>& src);

		void set(const uint i, const uint j, const T value);
		T get(const uint i, const uint j);

	private:
		T* m_data = nullptr;
		uint m_nx = 0;
		uint m_ny = 0;
		uint m_pitch = 0;
	};

	template<typename T>
	using DArray2D = Array2D<T, DeviceType::GPU>;

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
		if (m_nx != src.nx() || m_ny != src.ny()){
			this->resize(src.nx(), src.ny());
		}

		cuSafeCall(cudaMemcpy2D(m_data, m_pitch, src.begin(), src.pitch(), sizeof(T) * src.nx(), src.ny(), cudaMemcpyDeviceToDevice));
	}

	template<typename T>
	void Array2D<T, DeviceType::GPU>::assign(const Array2D<T, DeviceType::CPU>& src)
	{
		if (m_nx != src.nx() || m_ny != src.ny()) {
			this->resize(src.nx(), src.ny());
		}

		cuSafeCall(cudaMemcpy2D(m_data, m_pitch, src.begin(), sizeof(T) *src.nx(), sizeof(T) * src.nx(), src.ny(), cudaMemcpyHostToDevice));
	}

	template<typename T>
	void Array2D<T, DeviceType::GPU>::set(const uint i, const uint j, const T value)
	{
		assert(i < m_nx && j < m_ny && i >= 0 && j >= 0);
		
		char* addr = (char*)m_data;
		addr += j * m_pitch;

		cuSafeCall(cudaMemcpy(&((T*)addr)[i], &value, sizeof(T), cudaMemcpyHostToDevice));
	}

	template<typename T>
	T Array2D<T, DeviceType::GPU>::get(const uint i, const uint j)
	{
		assert(i < m_nx && j < m_ny && i >= 0 && j >= 0);

		char* addr = (char*)m_data;
		addr += j * m_pitch;

		T value;

		cuSafeCall(cudaMemcpy(&value, &((T*)addr)[i], sizeof(T), cudaMemcpyDeviceToHost));

		return value;
	}
}
