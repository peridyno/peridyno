#include "ArrayList.h"
#include "Catalyzer/VkScan.h"
#include "Catalyzer/VkReduce.h"

namespace dyno {
	template<typename T>
	CArrayList<T>::CArrayList()
	{

	}

	template<typename T>
	void CArrayList<T>::resize(const std::vector<uint32_t>& num)
	{
		mIndex.resize(num.size());
		uint32_t accNum = 0;
		for (size_t i = 0; i < num.size(); i++)
		{
			mIndex[i] = accNum;
			accNum += num[i];
		}

		mElements.resize(accNum);

		mInfo.arraySize = num.size();
		mInfo.totalSize = accNum;
	}

	template<typename T>
	DArrayList<T>::DArrayList()
	{

	}

	template<typename T>
	void DArrayList<T>::resize(const std::vector<uint32_t>& num)
	{
		VkDeviceArray<uint32_t> deviceNum;
		deviceNum.resize(num.size());

		vkTransfer(deviceNum, num);

		this->resize(deviceNum);

		deviceNum.clear();
	}

	template<typename T>
	void DArrayList<T>::resize(const VkDeviceArray<uint32_t>& num)
	{
		mIndex.resize(num.size());

		VkScan<uint32_t> mScan;
		VkReduce<uint32_t> mReduce;

		mScan.scan(mIndex, num, EXCLUSIVESCAN);

		uint32_t totalNum = mReduce.reduce(num);

		mElements.resize(totalNum);

		ArrayListInfo info;
		info.arraySize = num.size();
		info.totalSize = totalNum;

		mInfo.setValue(info);
	}
}
