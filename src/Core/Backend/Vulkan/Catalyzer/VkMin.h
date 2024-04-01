#pragma once
#include "VkDeviceArray.h"
#include "VkProgram.h"

using uint = unsigned int;
namespace dyno {
	/*! TODO:
	*  \brief implement functions for reducing a range to a single value
	*/
	// T suport int, float and uint32_t types
	template<typename T>
	class VkMin {
		
	public:
		VkMin();
		~VkMin();

		T reduce(const std::vector<T>& input);
		T reduce(const VkDeviceArray<T>& input);

	private:
		std::shared_ptr<VkProgram> mReduceKernel;
	};

	template<typename T>
	class VkMinElement {
		
	public:
		VkMinElement();
		~VkMinElement();

		uint reduce(const std::vector<T>& input);
		uint reduce(const VkDeviceArray<T>& input);

	private:
		std::shared_ptr<VkProgram> mReduceKernel;
	};
}
#include "VkMin.inl"
