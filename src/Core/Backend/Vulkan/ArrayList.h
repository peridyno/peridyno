#pragma once
#include "VkDeviceArray.h"
#include "VkUniform.h"

namespace dyno {

	struct ArrayListInfo
	{
		uint32_t arraySize = 0;
		uint32_t totalSize = 0;
	};

	template<typename T>
	class CArrayList
	{
	public:
		CArrayList();
		~CArrayList() {};

		void resize(const std::vector<uint32_t>& num);

		uint32_t size(uint32_t id)
		{
			return id == mInfo.arraySize - 1 ? mInfo.arraySize - mIndex[id] : mIndex[id + 1] - mIndex[id];
		}

	public:
		std::vector<uint32_t> mIndex;
		std::vector<T> mElements;

		ArrayListInfo mInfo;
	};

	template<typename T>
	class DArrayList
	{
	public:
		DArrayList();

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~DArrayList() {};

		void resize(const std::vector<uint32_t>& num);
		void resize(const VkDeviceArray<uint32_t>& num);
		
	public:
		VkDeviceArray<uint32_t> mIndex;
		VkDeviceArray<T> mElements;

		VkUniform<ArrayListInfo> mInfo;
	};
}

#include "ArrayList.inl"
