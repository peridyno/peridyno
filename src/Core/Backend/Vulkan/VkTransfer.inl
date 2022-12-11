#include <assert.h>
#include "VkContext.h"

namespace dyno
{
	template<typename T>
	bool vkTransfer(VkHostArray<T>& dst, const VkDeviceArray<T>& src)
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
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.currentContext() == src.currentContext());
		assert(dst.size() == src.size());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size()*sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}

	template<typename T>
	bool vkTransfer(std::vector<T>& dst, const VkDeviceArray<T>& src)
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
	bool vkTransfer(VkDeviceArray<T>& dst, const std::vector<T>& src)
	{
		VkContext* ctx = dst.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

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
	bool vkTransfer(VkDeviceArray<T>& dst, const VkDeviceArray<T>& src)
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
	bool vkTransfer(VkDeviceArray<T>& dst, uint64_t dstOffset, const VkDeviceArray<T>& src, uint64_t srcOffset, uint64_t copySize)
	{
		if (copySize <= 0)
			return false;

		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.currentContext() == src.currentContext());
		assert(dst.size() >= dstOffset + copySize);
		assert(src.size() >= srcOffset + copySize);

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.dstOffset = dstOffset * sizeof(T);
		copyRegion.srcOffset = srcOffset * sizeof(T);
		copyRegion.size = src.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
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