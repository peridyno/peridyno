#pragma once
#include "VkDeviceArray.h"
#include "VkProgram.h"
#include "VkConstant.h"

#define EXCLUSIVESCAN   0 
#define INCLUSIVESCAN   1 

using uint = unsigned int;
namespace dyno {
	/*! TODO:
	*  \brief implement functions for computing prefix sums
	*/
	//suport int float and uint32_t 
	struct ScanParameters {
		int n;
		int ScanType;
	};

	template<typename T>
	class VkScan{
	public :
		VkScan();
		~VkScan();

		enum Type {
			Exclusive = 0,
			Inclusive,
		};

		void scan(std::vector<T>& input, uint ScanType);

		void scan(VkDeviceArray<T>& output, const VkDeviceArray<T>& input, uint ScanType);
	private:
		std::shared_ptr<VkProgram> mScan;

		std::shared_ptr<VkProgram> mAdd;
	};
}
#include "VkScan.inl"