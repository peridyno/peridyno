#include "VkProgram.h"
#include "VulkanTools.h"

#include <assert.h>

namespace dyno
{
	VkProgram::~VkProgram()
	{
		ctx->descriptorCache().releaseLayoutHandle(mLayoutHandle);

		for (auto shderModule : shaderModules) {
			vkDestroyShaderModule(ctx->deviceHandle(), shderModule, nullptr);
		}

		vkDestroyPipeline(ctx->deviceHandle(), pipeline, nullptr);
		vkDestroyPipelineLayout(ctx->deviceHandle(), pipelineLayout, nullptr);

		vkDestroySemaphore(ctx->deviceHandle(), compute.semaphores.ready, nullptr);
		vkDestroySemaphore(ctx->deviceHandle(), compute.semaphores.complete, nullptr);
	}

	VkCommandBuffer VkProgram::commandBuffer() const {
		if (mCommandBuffer) {
			return mCommandBuffer.value();
		}
		return  VkCompContext::current().commandBuffer();
	}

	void VkProgram::begin()
	{
		//if(descriptorPool != VK_NULL_HANDLE) {
		//	vkResetDescriptorPool(ctx->deviceHandle(), descriptorPool, 0);
		//}
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		cmdBufInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

		VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer(), &cmdBufInfo));
		//vkCmdBindPipeline(mCommandBuffers, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	}

	/*
	void VkProgram::dispatch(dim3 groupSize)
	{
		//vkCmdBindDescriptorSets(mCommandBuffers, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, descriptorSet ? 1 : 0, &descriptorSet, 0, 0);
		auto cmd = commandBuffer();
		auto pushBuf = buildPushBuf(mConstArgs);
		vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBuf.size(), pushBuf.data());

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		vkCmdDispatch(cmd, groupSize.x, groupSize.y, groupSize.z);

		addComputeToComputeBarriers(cmd);
	}
	*/

	void VkProgram::end()
	{
		auto cmd = commandBuffer();
		vkEndCommandBuffer(cmd);
	}

	auto VkProgram::update(bool sync) -> std::optional<VkFence>
	{
		auto cmd = commandBuffer();

		static bool firstDraw = true;
		VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		// FIXME find a better way to do this (without using fences, which is much slower)
		VkPipelineStageFlags computeWaitDstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		if (!firstDraw) {
			// TODO: semaphore should use in different submit queue.
			//computeSubmitInfo.waitSemaphoreCount = 1;
			//computeSubmitInfo.pWaitSemaphores = &compute.semaphores.ready;
			//computeSubmitInfo.pWaitDstStageMask = &computeWaitDstStageMask;
		}
		else {
			firstDraw = false;
		}
		//computeSubmitInfo.signalSemaphoreCount = 1;
		//computeSubmitInfo.pSignalSemaphores = &compute.semaphores.complete;
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &cmd;

		if (sync) {
			VkFence fence {VK_NULL_HANDLE};
			VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
			VK_CHECK_RESULT(vkCreateFence(ctx->deviceHandle(), &fenceInfo, nullptr, &fence));
			VkCompContext::current().submit(queue, 1, &computeSubmitInfo, fence);
			return fence;
		}
		else {
			VkCompContext::current().submit(queue, 1, &computeSubmitInfo, VK_NULL_HANDLE);
		}
		return std::nullopt;
	}

	void VkProgram::wait(VkFence fence)
	{
		VK_CHECK_RESULT(vkWaitForFences(ctx->deviceHandle(), 1, &fence, VK_TRUE, UINT64_MAX));
		vkDestroyFence(ctx->deviceHandle(), fence, nullptr);
	}

	void VkProgram::addGraphicsToComputeBarriers(VkCommandBuffer commandBuffer, std::vector<VkVariable*> bufferArgs)
	{
		if (ctx->isComputeQueueSpecial()) {
			VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
			bufferBarrier.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			bufferBarrier.srcQueueFamilyIndex = ctx->graphicsQueueFamilyIndex();
			bufferBarrier.dstQueueFamilyIndex = ctx->computeQueueFamilyIndex();
			bufferBarrier.size = VK_WHOLE_SIZE;

			std::vector<VkBufferMemoryBarrier> bufferBarriers;
			for (size_t i = 0; i < bufferArgs.size(); i++)
			{
				bufferBarrier.buffer = bufferArgs[i]->bufferHandle();
				bufferBarriers.push_back(bufferBarrier);
			}
			// 			bufferBarrier.buffer = input->bufferHandle();
			// 			bufferBarriers.push_back(bufferBarrier);
			// 			bufferBarrier.buffer = output->bufferHandle();
			// 			bufferBarriers.push_back(bufferBarrier);
			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),
				0, nullptr);
		}
	}

	void VkProgram::addComputeToComputeBarriers(VkCommandBuffer commandBuffer, std::vector<VkVariable*> bufferArgs)
	{
		VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
		bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		bufferBarrier.srcQueueFamilyIndex = ctx->computeQueueFamilyIndex();
		bufferBarrier.dstQueueFamilyIndex = ctx->computeQueueFamilyIndex();
		bufferBarrier.size = VK_WHOLE_SIZE;
		std::vector<VkBufferMemoryBarrier> bufferBarriers;
		for (size_t i = 0; i < bufferArgs.size(); i++)
		{
			auto handle = bufferArgs[i]->bufferHandle();
			// bypass null handle
			if(handle != VK_NULL_HANDLE) {
				bufferBarrier.buffer = handle;
				bufferBarriers.push_back(bufferBarrier);
			}
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_FLAGS_NONE,
			0, nullptr,
			static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),
			0, nullptr);
	}

	void VkProgram::addComputeToGraphicsBarriers(VkCommandBuffer commandBuffer, std::vector<VkVariable*> bufferArgs)
	{
		if (ctx->isComputeQueueSpecial()) {
			VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
			bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
			bufferBarrier.srcQueueFamilyIndex = ctx->computeQueueFamilyIndex();
			bufferBarrier.dstQueueFamilyIndex = ctx->graphicsQueueFamilyIndex();
			bufferBarrier.size = VK_WHOLE_SIZE;
			std::vector<VkBufferMemoryBarrier> bufferBarriers;
			for (size_t i = 0; i < bufferArgs.size(); i++)
			{
				bufferBarrier.buffer = bufferArgs[i]->bufferHandle();
				bufferBarriers.push_back(bufferBarrier);
			}
			// 			std::vector<VkBufferMemoryBarrier> bufferBarriers;
			// 			bufferBarrier.buffer = input->bufferHandle();
			// 			bufferBarriers.push_back(bufferBarrier);
			// 			bufferBarrier.buffer = output->bufferHandle();
			// 			bufferBarriers.push_back(bufferBarrier);
			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),
				0, nullptr);
		}
	}


	void VkProgram::bindPipeline()
	{
		auto cmd = VkCompContext::current().commandBuffer();
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	}


	void VkProgram::setVkCommandBuffer(VkCommandBuffer cmdBuffer) {
		mCommandBuffer = cmdBuffer;
	}

	void VkProgram::suspendInherentCmdBuffer(VkCommandBuffer cmdBuffer)
	{
		mOldCommandBuffer = mCommandBuffer;
		mCommandBuffer = cmdBuffer;
	}

	void VkProgram::restoreInherentCmdBuffer()
	{
		mCommandBuffer = mOldCommandBuffer;
	}

	bool VkProgram::load(std::filesystem::path fileName_) 
	{
		auto fileName = fileName_.string();
		//Create pipeline layout
		std::vector<VkPushConstantRange> pushConstantRanges;
		{
			auto range = buildPushRange(VK_SHADER_STAGE_COMPUTE_BIT, mFormalConstants);
			if(range.size > 0) pushConstantRanges.push_back(range);
		}

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		pipelineLayoutCreateInfo.pushConstantRangeCount = pushConstantRanges.size();
		pipelineLayoutCreateInfo.pPushConstantRanges = pushConstantRanges.data();

		VK_CHECK_RESULT(vkCreatePipelineLayout(ctx->deviceHandle(), &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		// Create pipeline
		VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(pipelineLayout, 0);
		computePipelineCreateInfo.stage = this->createComputeStage(fileName);
		VK_CHECK_RESULT(vkCreateComputePipelines(ctx->deviceHandle(), ctx->pipelineCacheHandle(), 1, &computePipelineCreateInfo, nullptr, &pipeline));

		return true;
	}

	VkPipelineShaderStageCreateInfo VkProgram::createComputeStage(std::string fileName)
	{
		VkPipelineShaderStageCreateInfo shaderStage = {};
		shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStage.module = vks::tools::loadShaderModule(fileName, ctx->deviceHandle());
		shaderStage.pName = "main";
		assert(shaderStage.module != VK_NULL_HANDLE);
		shaderModules.push_back(shaderStage.module);
		return shaderStage;
	}

	void VkProgram::pushFormalParameter(const VkArgInfo& arg)
	{
		mFormalParamters.push_back(arg);
	}

	void VkProgram::pushFormalConstant(const VkArgInfo& arg)
	{
		mFormalConstants.push_back(arg);
		
		this->pushFormalParameter(arg);
	}

	/*
	void VkProgram::pushArgument(VkVariable* arg)
	{
		mAllArgs.push_back(arg);
	}

	void VkProgram::pushConstant(VkVariable* arg)
	{
		assert(arg != nullptr && arg->type() == VariableType::Constant);

		mConstArgs.push_back(arg);
		this->pushArgument(arg);
	}

	void VkProgram::pushDeviceBuffer(VkVariable* arg)
	{
		assert(arg != nullptr && arg->type() == VariableType::DeviceBuffer);

		mBufferArgs.push_back(arg);
		this->pushArgument(arg);
	}

	void VkProgram::pushUniform(VkVariable* arg)
	{
		assert(arg != nullptr && arg->type() == VariableType::Uniform);

		mUniformArgs.push_back(arg);
		this->pushArgument(arg);
	}
	*/

	VkPushConstantRange VkProgram::buildPushRange(VkShaderStageFlags stage, const std::vector<VkArgInfo>& vars) {
		VkPushConstantRange range {};
		range.stageFlags = stage;
		range.offset = 0;
		for (size_t i = 0, offset = 0; i < vars.size(); i++)
		{
			auto size = vars[i].var_size;
			if(i > 0) {
				auto pre_size = vars[i-1].var_size;
				offset = vks::tools::alignedSize(std::max(offset + pre_size, (std::size_t)size), 4);
			}
			range.size = offset + vks::tools::alignedSize(size, 4); 
		}
		assert(range.size % 4 == 0);
		return range;
	}

	std::vector<std::byte> VkProgram::buildPushBuf(const std::vector<VkVariable*>& vars) {
		std::vector<std::byte> buf;
		for (size_t i = 0, offset = 0; i < vars.size(); i++)
		{
			const auto& varInfo = mFormalConstants.at(i);
			auto data = (std::byte*)vars[i]->data();
			auto size = vars[i]->bufferSize();
			assert(size == varInfo.var_size);
			auto align = varInfo.var_align;
			auto alignSize = vks::tools::alignedSize(size, align);
			if(i > 0) {
				auto pre_size = vars[i-1]->bufferSize();
				//vks::tools::alignedSize(std::max<std::size_t>(offset + pre_size, std::min<std::size_t>(16, size)), 4);
				offset = vks::tools::alignedSize(offset + pre_size, align);
			}
			buf.resize(offset + alignSize);
			std::copy(data, data + size, buf.begin() + offset);
		}
		assert(buf.size() % 4 == 0);
		return buf;
	}
}