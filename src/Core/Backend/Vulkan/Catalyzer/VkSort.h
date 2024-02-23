#pragma once
#include "Array/Array.h"
#include "VkDeviceArray.h"
#include "VkProgram.h"

namespace dyno {
	struct SortParam {
		enum eAlgorithmVariant : uint32_t {
			eLocalBms = 0,
			eLocalDisperse,
			eBigFlip,
			eBigDisperse
		};
		enum SortType : uint32_t {
			eUp = 0,
			eDown,
		};
	};

	template<typename T>
	class VkSort {
	public:
		VkSort(std::filesystem::path spv = {});
		~VkSort();

		// suport int、 float and uint32_t types --- SortType = UP /DOWN
		void sort(CArray<T>& data, uint32_t SortType);
		void sort(DArray<T> data, uint32_t SortType);
		void sort(VkDeviceArray<T> data, uint32_t SortType);

	private:
		std::shared_ptr<VkProgram> mSortKernel;
	};

	template<typename TKey, typename TVal = TKey>
	class VkSortByKey {
	public:
		VkSortByKey(std::filesystem::path spv = {});
		~VkSortByKey();

		void sortByKey(CArray<TKey>& key, CArray<TVal>& val, uint32_t SortType);
		void sortByKey(DArray<TKey> key, DArray<TVal> val, uint32_t SortType);
		void sortByKey(VkDeviceArray<TKey> key, VkDeviceArray<TVal> val, uint32_t SortType);
	private:
		std::shared_ptr<VkProgram> mSortKernel;
	};
}
#include "VkSort.inl"
