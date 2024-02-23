#pragma once
#include "VkHostArray.h"
#include "VkHostArray2D.h"
#include "VkDeviceArray.h"
#include "VkDeviceArray2D.h"
#include "VkDeviceArray3D.h"

namespace dyno 
{
	template<typename T>
	bool vkFill(VkDeviceArray<T>, uint64_t offset, uint64_t size, uint32_t data);

	template<typename T>
	bool vkFill(VkDeviceArray<T>, uint32_t data);

	template<typename T>
	bool vkTransfer(VkHostArray<T>& dst, uint64_t dstOffset, const VkDeviceArray<T>& src, uint64_t srcOffset, uint64_t size);
	template<typename T>
	bool vkTransfer(std::vector<T>& dst, uint64_t dstOffset, const VkDeviceArray<T>& src, uint64_t srcOffset, uint64_t size);
	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, uint64_t dstOffset, const VkDeviceArray<T>& src, uint64_t srcOffset, uint64_t size);
	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, uint64_t dstOffset, const VkHostArray<T>& src, uint64_t srcOffset, uint64_t size);
	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, uint64_t dstOffset, const std::vector<T>& src, uint64_t srcOffset, uint64_t size);

	template<typename T>
	bool vkTransfer(VkHostArray<T>& dst, const VkDeviceArray<T>& src);

	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, const VkHostArray<T>& src);

	template<typename T>
	bool vkTransfer(std::vector<T>& dst, const VkDeviceArray<T>& src);

	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, const std::vector<T>& src);

	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, const VkDeviceArray<T>& src);


	template<typename T>
	bool vkTransfer(VkHostArray<T>& dst, const VkDeviceArray2D<T>& src);

	template<typename T>
	bool vkTransfer(std::vector<T>& dst, const VkDeviceArray2D<T>& src);

	template<typename T>
	bool vkTransfer(VkDeviceArray2D<T>& dst, const std::vector<T>& src);

	template<typename T>
	bool vkTransfer(VkDeviceArray2D<T>& dst, const VkDeviceArray2D<T>& src);

	template<typename T>
	bool vkTransfer(VkDeviceArray3D<T>& dst, const std::vector<T>& src);

	template<typename T>
	bool vkTransfer(VkDeviceArray3D<T>& dst, const VkDeviceArray3D<T>& src);



}

#include "VkTransfer.inl"