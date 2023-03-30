namespace dyno 
{
	template<typename T>
	void Array<T, DeviceType::CPU>::assign(const Array<T, DeviceType::GPU>& src)
	{
		if (mData.size() != src.size())
			this->resize(src.size());

		cuSafeCall(cudaMemcpy(this->begin(), src.begin(), src.size() * sizeof(T), cudaMemcpyDeviceToHost));
	}

	/*!
	*	\class	Array
	*	\brief	This class is designed to be elegant, so it can be directly passed to GPU as parameters.
	*/
	template<typename T>
	class Array<T, DeviceType::GPU>
	{
	public:
		Array()
		{
		};

		Array(uint num)
		{
			this->resize(num);
		}

		/*!
		*	\brief	Do not release memory here, call clear() explicitly.
		*/
		~Array() {};

		void resize(const uint n);

		/*!
		*	\brief	Clear all data to zero.
		*/
		void reset();

		/*!
		*	\brief	Free allocated memory.	Should be called before the object is deleted.
		*/
		void clear();

		DYN_FUNC inline const T*	begin() const { return mData; }
		DYN_FUNC inline T*	begin() { return mData; }

		DeviceType	deviceType() { return DeviceType::GPU; }

		GPU_FUNC inline T& operator [] (unsigned int id) {
			return mData[id];
		}

		GPU_FUNC inline T& operator [] (unsigned int id) const {
			return mData[id];
		}

		DYN_FUNC inline uint size() const { return mTotalNum; }
		DYN_FUNC inline bool isCPU() const { return false; }
		DYN_FUNC inline bool isGPU() const { return true; }
		DYN_FUNC inline bool isEmpty() const { return mData == nullptr; }

		void assign(const Array<T, DeviceType::GPU>& src);
		void assign(const Array<T, DeviceType::CPU>& src);
		void assign(const std::vector<T>& src);

		void assign(const Array<T, DeviceType::GPU>& src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
		void assign(const Array<T, DeviceType::CPU>& src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
		void assign(const std::vector<T>& src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);

		friend std::ostream& operator<<(std::ostream &out, const Array<T, DeviceType::GPU>& dArray)
		{
			Array<T, DeviceType::CPU> hArray;
			hArray.assign(dArray);

			out << hArray;

			return out;
		}

	private:
		T* mData = nullptr;
		uint mTotalNum = 0;
		uint mBufferNum = 0;
	};
	
	template<typename T>
	using DArray = Array<T, DeviceType::GPU>;

	template<typename T>
	void Array<T, DeviceType::GPU>::resize(const uint n)
	{
		if (mTotalNum == n) return;

		if (n == 0) {
			clear();
			return;
		}

		int exp = std::ceil(std::log2(float(n)));

		int bound = std::pow(2, exp);

		if (n > mBufferNum || n <= mBufferNum / 2) {
			clear();

			mTotalNum = n; 	
			mBufferNum = bound;

			cuSafeCall(cudaMalloc(&mData, bound * sizeof(T)));
		}
		else
			mTotalNum = n;
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::clear()
	{
		if (mData != nullptr)
		{
			cuSafeCall(cudaFree((void*)mData));
		}

		mData = nullptr;
		mTotalNum = 0;
		mBufferNum = 0;
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::reset()
	{
		cuSafeCall(cudaMemset((void*)mData, 0, mTotalNum * sizeof(T)));
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::GPU>& src)
	{
		if (mTotalNum != src.size())
			this->resize(src.size());

		cuSafeCall(cudaMemcpy(mData, src.begin(), src.size() * sizeof(T), cudaMemcpyDeviceToDevice));
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::CPU>& src)
	{
		if (mTotalNum != src.size())
			this->resize(src.size());

		cuSafeCall(cudaMemcpy(mData, src.begin(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
	}


	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const std::vector<T>& src)
	{
		if (mTotalNum != src.size())
			this->resize((uint)src.size());

		cuSafeCall(cudaMemcpy(mData, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const std::vector<T>& src, const uint count, const uint dstOffset, const uint srcOffset)
	{
		cuSafeCall(cudaMemcpy(mData + dstOffset, src.data() + srcOffset, count * sizeof(T), cudaMemcpyHostToDevice));
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::CPU>& src, const uint count, const uint dstOffset, const uint srcOffset)
	{
		cuSafeCall(cudaMemcpy(mData + dstOffset, src.begin() + srcOffset, count * sizeof(T), cudaMemcpyHostToDevice));
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::GPU>& src, const uint count, const uint dstOffset, const uint srcOffset)
	{
		cuSafeCall(cudaMemcpy(mData + dstOffset, src.begin() + srcOffset, count * sizeof(T), cudaMemcpyDeviceToDevice));
	}
}
