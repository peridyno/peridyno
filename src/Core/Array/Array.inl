namespace dyno 
{
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
		cuSafeCall(cudaMemcpy(mData + dstOffset, src.begin() + srcOffset, count * sizeof(T), cudaMemcpyHostToDevice));
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

	template<typename T>
	void Array<T, DeviceType::CPU>::resize(const uint n)
	{
		mData.resize(n);
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::clear()
	{
		mData.clear();
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::reset()
	{
		memset((void*)mData.data(), 0, mData.size()*sizeof(T));
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::assign(const Array<T, DeviceType::GPU>& src)
	{
		if (mData.size() != src.size())
			this->resize(src.size());

		cuSafeCall(cudaMemcpy(this->begin(), src.begin(), src.size() * sizeof(T), cudaMemcpyDeviceToHost));
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::assign(const Array<T, DeviceType::CPU>& src)
	{
		if (mData.size() != src.size())
			this->resize(src.size());

		memcpy(this->begin(), src.begin(), src.size() * sizeof(T));
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::assign(const T& val)
	{
		mData.assign(mData.size(), val);
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::assign(uint num, const T& val)
	{
		mData.assign(num, val);
	}
}
