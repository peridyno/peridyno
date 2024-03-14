#include <assert.h>
#include "VkContext.h"
#include "VkCompContext.h"

namespace dyno
{
	namespace details 
	{
		inline bool vkTransfer(VkContext* ctx, VkBuffer dst, uint64_t dstOffset, VkBuffer src, uint64_t srcOffset, uint64_t size) {
			VkBufferCopy copyRegion = {};
			copyRegion.dstOffset = dstOffset;
			copyRegion.srcOffset = srcOffset;
			copyRegion.size = size;

			auto& comp = VkCompContext::current();
			if (!comp.isDelaySubmit()) {
				VkCommandBuffer copyCmd = comp.commandBuffer(true);
				VkCommandBufferBeginInfo cmdBeginInfo = vks::initializers::commandBufferBeginInfo();
				VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &cmdBeginInfo));
				// VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
				vkCmdCopyBuffer(copyCmd, src, dst, 1, &copyRegion);
				ctx->flushCommandBuffer(copyCmd, ctx->computeQueueHandle(), false);
				comp.releaseCommandBuffer(copyCmd);
			}
			else {
				VkCommandBuffer cmd = comp.commandBuffer(true);
				VkCommandBufferBeginInfo cmdBeginInfo = vks::initializers::commandBufferBeginInfo();
				VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
				vkCmdCopyBuffer(cmd, src, dst, 1, &copyRegion);
				comp.addTransferBarrier(cmd, dst);
				VK_CHECK_RESULT(vkEndCommandBuffer(cmd));
				VkSubmitInfo submitInfo = vks::initializers::submitInfo();
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &cmd;
				comp.submit(ctx->computeQueueHandle(), 1, &submitInfo, VK_NULL_HANDLE);
			}
			return true;
		}

		template<template<class> class VKARR_A, template<class> class VKARR_B, typename T>
		bool vkTransfer(VKARR_A<T>& dst, uint64_t dstOffset, const VKARR_B<T>& src, uint64_t srcOffset, uint64_t size) {
			VkContext* ctx = src.currentContext();

			assert(ctx != nullptr);
			assert(dst.currentContext() == src.currentContext());
			assert(dst.size() >= dstOffset + size);
			assert(src.size() >= srcOffset + size);

			if(size == 0) return true;
			return vkTransfer(ctx, dst.bufferHandle(), dstOffset * sizeof(T), src.bufferHandle(), srcOffset * sizeof(T), size * sizeof(T));
		}
	};

	template<typename T>
	bool vkFill(VkDeviceArray<T> dst, uint64_t offset, uint64_t size, uint32_t data) {
			VkContext* ctx = dst.currentContext();
			assert(ctx != nullptr);
			assert(dst.size() >= offset + size);
			if(size == 0) return true;

			auto vk_offset = offset * sizeof(T);
			auto vk_size = size * sizeof(T);

			// Copy from staging buffer
			auto& comp = VkCompContext::current();
			if (!comp.isDelaySubmit()) {
				VkCommandBuffer cmd = comp.commandBuffer(true);
				VkCommandBufferBeginInfo cmdBeginInfo = vks::initializers::commandBufferBeginInfo();
				VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
				// VkCommandBuffer cmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
				vkCmdFillBuffer(cmd, dst.bufferHandle(), vk_offset, vk_size, data);
				ctx->flushCommandBuffer(cmd, ctx->computeQueueHandle(), false);
				comp.releaseCommandBuffer(cmd);
			}
			else {
				VkCommandBuffer cmd = comp.commandBuffer(true);
				VkCommandBufferBeginInfo cmdBeginInfo = vks::initializers::commandBufferBeginInfo();
				VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
				vkCmdFillBuffer(cmd, dst.bufferHandle(), vk_offset, vk_size, data);
				comp.addTransferBarrier(cmd, dst.bufferHandle());
				VK_CHECK_RESULT(vkEndCommandBuffer(cmd));
				VkSubmitInfo submitInfo = vks::initializers::submitInfo();
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &cmd;
				comp.submit(ctx->computeQueueHandle(), 1, &submitInfo, VK_NULL_HANDLE);
			}
			return true;
	}

	template<typename T>
	bool vkFill(VkDeviceArray<T> dst, uint32_t data) {
		return vkFill(dst, 0, dst.size(), data);
	}

	template<typename T>
	bool vkTransfer(VkHostArray<T>& dst, uint64_t dstOffset, const VkDeviceArray<T>& src, uint64_t srcOffset, uint64_t size)
	{
		return details::vkTransfer(dst, dstOffset, src, srcOffset, size);
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, uint64_t dstOffset, const VkDeviceArray<T>& src, uint64_t srcOffset, uint64_t size)
	{
		return details::vkTransfer(dst, dstOffset, src, srcOffset, size);
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, uint64_t dstOffset, const VkHostArray<T>& src, uint64_t srcOffset, uint64_t size) {
		return details::vkTransfer(dst, dstOffset, src, srcOffset, size);
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, uint64_t dstOffset, const std::vector<T>& src, uint64_t srcOffset, uint64_t size) {
		if(size == 0) return true;

		VkHostArray<T> vkHostSrc;
		vkHostSrc.resize(size);
		memcpy(vkHostSrc.mapped(), src.data() + srcOffset, sizeof(T)*size);
		vkTransfer(dst, 0, vkHostSrc, 0, size);
		vkHostSrc.clear();
		return true;
	}

		template<typename T>
	bool vkTransfer(std::vector<T>& dst, uint64_t dstOffset, const VkDeviceArray<T>& src, uint64_t srcOffset, uint64_t size)
	{
		if(size == 0) return true;

		VkHostArray<T> vkHostSrc;
		vkHostSrc.resize(size);
		vkTransfer(vkHostSrc, 0, src, srcOffset, size);
		memcpy(dst.data() + dstOffset, vkHostSrc.mapped(), sizeof(T)*size);
		vkHostSrc.clear();
		return true;
	}

	template<typename T>
	bool vkTransfer(VkHostArray<T>& dst, const VkDeviceArray<T>& src)
	{
		return vkTransfer(dst, 0, src, 0, src.size());
	}

	template<typename T>
	bool vkTransfer(VkHostArray<T>& dst, const VkDeviceArray2D<T>& src)
	{
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.currentContext() == src.currentContext());
		assert(dst.size() == src.size());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		/*		VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
				bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				bufferBarrier.srcQueueFamilyIndex = ctx->queueFamilyIndices.compute;
				bufferBarrier.dstQueueFamilyIndex = ctx->queueFamilyIndices.graphics;
				bufferBarrier.size = VK_WHOLE_SIZE;
				bufferBarrier.buffer = dst.bufferHandle();
				std::vector<VkBufferMemoryBarrier> bufferBarriers;
				bufferBarriers.push_back(bufferBarrier);

				vkCmdPipelineBarrier(
					copyCmd,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_FLAGS_NONE,
					0, nullptr,
					static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),
					0, nullptr);*/

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}

// 	VkBufferMemoryBarrier barrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
// 	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
// 	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
// 	barrier.buffer = pair.dst;
// 	barrier.size = pair.src.m_size;
// 	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, 0, 1, &barrier, 0, 0);


	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, const VkHostArray<T>& src)
	{
		return vkTransfer(dst, 0, src, 0, src.size());
	}

	template<typename T>
	bool vkTransfer(std::vector<T>& dst, const VkDeviceArray<T>& src)
	{
		return vkTransfer(dst, 0, src, 0, src.size());
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, const std::vector<T>& src)
	{
		return vkTransfer(dst, 0, src, 0, src.size());
	}


	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, const VkDeviceArray<T>& src)
	{
		return vkTransfer(dst, 0, src, 0, src.size());
	}



	template<typename T>
	bool vkTransfer(std::vector<T>& dst, const VkDeviceArray2D<T>& src)
	{
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		VkHostArray<T> vkHostSrc;
		vkHostSrc.resize(src.size());

		vkTransfer(vkHostSrc, src);

		memcpy(dst.data(), vkHostSrc.mapped(), sizeof(T)*src.size());

		vkHostSrc.clear();

		return true;
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray2D<T>& dst, const std::vector<T>& src)
	{
		VkContext* ctx = dst.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		if(src.size() == 0) return true;
		if(dst.bufferHandle() == VK_NULL_HANDLE) return false;

		VkHostArray<T> vkHostSrc;
		vkHostSrc.resize(src.size(), src.data());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, vkHostSrc.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		vkHostSrc.clear();

		return true;
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray2D<T>& dst, const VkDeviceArray2D<T>& src)
	{
		VkContext* ctx = dst.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}
	

	template<typename T>
	bool vkTransfer(VkDeviceArray3D<T>& dst, const std::vector<T>& src)
	{
		VkContext* ctx = dst.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		if(src.size() == 0) return true;

		VkHostArray<T> vkHostSrc;
		vkHostSrc.resize(src.size(), src.data());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, vkHostSrc.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		vkHostSrc.clear();

		return true;
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray3D<T>& dst, const VkDeviceArray3D<T>& src)
	{
		VkContext* ctx = dst.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}

	template<typename T>
	bool vkTransfer(VkHostArray<T>& dst, const VkDeviceArray3D<T>& src)
	{
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.currentContext() == src.currentContext());
		assert(dst.size() == src.size());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}

	template<typename T>
	bool vkTransfer(std::vector<T>& dst, const VkDeviceArray3D<T>& src)
	{
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		VkHostArray<T> vkHostSrc;
		vkHostSrc.resize(src.size());

		vkTransfer(vkHostSrc, src);

		memcpy(dst.data(), vkHostSrc.mapped(), sizeof(T)*src.size());

		vkHostSrc.clear();

		return true;
	}


}