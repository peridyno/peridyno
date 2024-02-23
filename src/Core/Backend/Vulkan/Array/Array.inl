#include "VkDeviceArray.h"
#include "VkTransfer.h"

namespace dyno 
{
	template<typename T>
	void Array<T, DeviceType::CPU>::assign(const Array<T, DeviceType::GPU>& src)
	{
		if (mData.size() != src.size())
			this->resize(src.size());

		vkTransfer(mData, *src.handle());
		//cuSafeCall(cudaMemcpy(this->begin(), src.begin(), src.size() * sizeof(T), cudaMemcpyDeviceToHost));
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

		Array(VkDeviceArray<T> arr):mData(arr) {
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

 		inline VkDeviceArray<T> begin() const { return mData; }

		inline const VkDeviceArray<T>* handle() const { return &mData; }
		inline VkDeviceArray<T>* handle() { return &mData; }

		VkBuffer buffer() const { return mData.bufferHandle(); }

		uint32_t bufferSize() { return mData.bufferSize(); }

		DeviceType	deviceType() { return DeviceType::GPU; }

		 inline T& operator [] (unsigned int id) {
			return mData[id];
		}

		inline T& operator [] (unsigned int id) const {
			return mData[id];
		}

		inline uint size() const { return mData.size(); }
		inline bool isCPU() const { return false; }
		inline bool isGPU() const { return true; }
		inline bool isEmpty() const { return mData.size() == 0; }

		void assign(const Array<T, DeviceType::GPU>& src);
		void assign(const Array<T, DeviceType::CPU>& src);
		void assign(const std::vector<T>& src);

		void assign(const Array<T, DeviceType::GPU>& src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
		void assign(const Array<T, DeviceType::CPU>& src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
		void assign(const std::vector<T>& src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);

		friend std::ostream& operator<<(std::ostream& out, const Array<T, DeviceType::GPU>& dArray)
		{
			Array<T, DeviceType::CPU> hArray;
			hArray.assign(dArray);

			out << hArray;

			return out;
		}

	private:
		VkDeviceArray<T> mData;
	};

	template<typename T>
	using DArray = Array<T, DeviceType::GPU>;

	template<typename T>
	void Array<T, DeviceType::GPU>::resize(const uint n)
	{
		if (mData.size() == n) return;

		if (n == 0) {
			mData.clear();
			return;
		}

		mData.resize(n);
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::clear()
	{
		mData.clear();
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::reset()
	{
		vkFill(mData, 0);
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::GPU>& src)
	{
		if (src.size() == 0)
		{
			mData.clear();
			return;
		}

		if (mData.size() != src.size())
			this->resize(src.size());

		vkTransfer(mData, *src.handle());
		//cuSafeCall(cudaMemcpy(mData, src.begin(), src.size() * sizeof(T), cudaMemcpyDeviceToDevice));
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::CPU>& src)
	{
		if(src.size() == 0) {
			mData.clear();
			return;
		}
		if (mData.size() != src.size())
			this->resize(src.size());

		vkTransfer(mData, *src.handle());
	}


	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const std::vector<T>& src)
	{
		if(src.size() == 0) {
			mData.clear();
			return;
		}
		if (mData.size() != src.size())
			this->resize((uint)src.size());

		vkTransfer(mData, src);
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const std::vector<T>& src, const uint count, const uint dstOffset, const uint srcOffset)
	{
		vkTransfer(mData, (uint64_t)dstOffset, src, (uint64_t)srcOffset, (uint64_t)count);
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::CPU>& src, const uint count, const uint dstOffset, const uint srcOffset)
	{
		vkTransfer(mData, (uint64_t)dstOffset, *src.handle(), (uint64_t)srcOffset, (uint64_t)count);
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::GPU>& src, const uint count, const uint dstOffset, const uint srcOffset)
	{
		vkTransfer(mData, (uint64_t)dstOffset, *src.handle(), (uint64_t)srcOffset, (uint64_t)count);
	}
}
