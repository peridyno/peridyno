#pragma once
#include "VkDeviceArray.h"
#include "VkProgram.h"

//SortType
#define UP   0 //1,2,3,4,5
#define DOWN 1 //5,4,3,2,1
namespace dyno {
	/*! TODO:
	*  \brief implement functions for reorganizing ranges into sorted order
	*/

	struct Parameters {
		enum eAlgorithmVariant : uint32_t {
			eLocalBms = 0,
			eLocalDisperse = 1,
			eBigFlip = 2,
			eBigDisperse = 3,
			addZero = 4,
			subtractZero = 5,
		};
		uint32_t          h;
		uint32_t          SortType;
		uint32_t          srcSize;
		uint32_t          dstSize;
		eAlgorithmVariant algorithm;
	};

	template<typename T>
	class VkSort{
	public:
		VkSort();
		~VkSort();

		// suport int¡¢ float and uint32_t types --- SortType = UP /DOWN
		void sort(std::vector<T> &data, uint32_t SortType);
		void sort(VkDeviceArray<T>& data, uint32_t SortType);


		// only suport keys[int] and values[int] types --- SortType = UP /DOWN
		void sort_by_key(std::vector<T>& keys, std::vector<T>& values, uint32_t SortType);
		void sort_by_key(VkDeviceArray<T>& keys, VkDeviceArray<T>& values, uint32_t SortType);

	private:
		std::shared_ptr<VkProgram> mSortKernel;
		std::shared_ptr<VkProgram> mSortByKeyKernel;
	};
}
#include "VkSort.inl"
