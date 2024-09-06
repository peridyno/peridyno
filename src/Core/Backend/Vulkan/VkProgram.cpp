#include "VkProgram.h"
#include "VulkanTools.h"

#include <assert.h>

namespace dyno
{
	VkProgram::~VkProgram()
	{
		mAllArgs.clear();
		mBufferArgs.clear();
		mUniformArgs.clear();
		mConstArgs.clear();

		for (auto& shderModule : shaderModules) {
			vkDestroyShaderModule(ctx->deviceHandle(), shderModule, nullptr);
		}

		vkDestroyPipeline(ctx->deviceHandle(), pipeline, nullptr);
		vkDestroyPipelineLayout(ctx->deviceHandle(), pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(ctx->deviceHandle(), descriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(ctx->deviceHandle(), descriptorPool, nullptr);

		vkDestroyCommandPool(ctx->deviceHandle(), mCommandPool, nullptr);
		vkDestroyFence(ctx->deviceHandle(), mFence, nullptr);
		vkDestroySemaphore(ctx->deviceHandle(), compute.semaphores.ready, nullptr);
		vkDestroySemaphore(ctx->deviceHandle(), compute.semaphores.complete, nullptr);
	}

	void VkProgram::begin()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		cmdBufInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

		VK_CHECK_RESULT(vkBeginCommandBuffer(mCommandBuffers, &cmdBufInfo));
		//vkCmdBindPipeline(mCommandBuffers, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	}

	void VkProgram::dispatch(dim3 groupSize)
	{
		vkCmdBindDescriptorSets(mCommandBuffers, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, 0);
		uint32_t offset = 0;
		for (size_t i = 0; i < mAllArgs.size(); i++)
		{
			auto variable = mAllArgs[i];
			if (variable->type() == VariableType::Constant) {
				vkCmdPushConstants(mCommandBuffers, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, offset, variable->bufferSize(), variable->data());
				offset += variable->bufferSize();
			}
		}

		vkCmdBindPipeline(mCommandBuffers, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		vkCmdDispatch(mCommandBuffers, groupSize.x, groupSize.y, groupSize.z);

		addComputeToComputeBarriers(mCommandBuffers);
	}

	void VkProgram::end()
	{
		vkEndCommandBuffer(mCommandBuffers);
	}

	void VkProgram::update(bool sync)
	{
		vkResetFences(ctx->deviceHandle(), 1, &mFence);

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
		computeSubmitInfo.pCommandBuffers = &mCommandBuffers;

		if (sync) {
			VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &computeSubmitInfo, mFence));
		}
		else {
			VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));
		}
	}

	void VkProgram::wait()
	{
		VK_CHECK_RESULT(vkWaitForFences(ctx->deviceHandle(), 1, &mFence, VK_TRUE, UINT64_MAX));
	}

	void VkProgram::addGraphicsToComputeBarriers(VkCommandBuffer commandBuffer)
	{
		if (ctx->isComputeQueueSpecial()) {
			VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
			bufferBarrier.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			bufferBarrier.srcQueueFamilyIndex = ctx->queueFamilyIndices.graphics;
			bufferBarrier.dstQueueFamilyIndex = ctx->queueFamilyIndices.compute;
			bufferBarrier.size = VK_WHOLE_SIZE;

			std::vector<VkBufferMemoryBarrier> bufferBarriers;
			for (size_t i = 0; i < mBufferArgs.size(); i++)
			{
				bufferBarrier.buffer = mBufferArgs[i]->bufferHandle();
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

	void VkProgram::addComputeToComputeBarriers(VkCommandBuffer commandBuffer)
	{
		VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
		bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		bufferBarrier.srcQueueFamilyIndex = ctx->queueFamilyIndices.compute;
		bufferBarrier.dstQueueFamilyIndex = ctx->queueFamilyIndices.compute;
		bufferBarrier.size = VK_WHOLE_SIZE;
		std::vector<VkBufferMemoryBarrier> bufferBarriers;
		for (size_t i = 0; i < mBufferArgs.size(); i++)
		{
			bufferBarrier.buffer = mBufferArgs[i]->bufferHandle();
			bufferBarriers.push_back(bufferBarrier);
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

	void VkProgram::addComputeToGraphicsBarriers(VkCommandBuffer commandBuffer)
	{
		if (ctx->isComputeQueueSpecial()) {
			VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
			bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
			bufferBarrier.srcQueueFamilyIndex = ctx->queueFamilyIndices.compute;
			bufferBarrier.dstQueueFamilyIndex = ctx->queueFamilyIndices.graphics;
			bufferBarrier.size = VK_WHOLE_SIZE;
			std::vector<VkBufferMemoryBarrier> bufferBarriers;
			for (size_t i = 0; i < mBufferArgs.size(); i++)
			{
				bufferBarrier.buffer = mBufferArgs[i]->bufferHandle();
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
		vkCmdBindPipeline(mCommandBuffers, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	}

	void VkProgram::suspendInherentCmdBuffer(VkCommandBuffer cmdBuffer)
	{
		mCmdBufferCopy = mCommandBuffers;
		mCommandBuffers = cmdBuffer;
	}

	void VkProgram::restoreInherentCmdBuffer()
	{
		mCommandBuffers = mCmdBufferCopy;
	}

	bool VkProgram::load(std::string fileName)
	{
		//Create pipeline layout
		std::vector<VkPushConstantRange> pushConstantRanges;
		for (size_t i = 0; i < mFormalConstants.size(); i++)
		{
			pushConstantRanges.push_back(vks::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, mFormalConstants[i]->bufferSize(), i));
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

	void VkProgram::pushFormalParameter(VkVariable* arg)
	{
		mFormalParamters.push_back(arg);
	}

	void VkProgram::pushFormalConstant(VkVariable* arg)
	{
		mFormalConstants.push_back(arg);
		
		this->pushFormalParameter(arg);
	}

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

	VkMultiProgram::VkMultiProgram()
	{
		auto inst = VkSystem::instance();
		ctx = VkSystem::instance()->currentContext();

		// Create the command pool
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = ctx->queueFamilyIndices.compute;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(ctx->deviceHandle(), &cmdPoolInfo, nullptr, &commandPool));

		// Create a command buffer for compute operations
		VkCommandBufferAllocateInfo cmdBufAllocateInfo =
			vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);

		VK_CHECK_RESULT(vkAllocateCommandBuffers(ctx->deviceHandle(), &cmdBufAllocateInfo, &commandBuffers));

		VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
		VK_CHECK_RESULT(vkCreateFence(ctx->deviceHandle(), &fenceInfo, nullptr, &mFence));

		// Semaphores for graphics / compute synchronization
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(ctx->deviceHandle(), &semaphoreCreateInfo, nullptr, &compute.semaphores.ready));
		VK_CHECK_RESULT(vkCreateSemaphore(ctx->deviceHandle(), &semaphoreCreateInfo, nullptr, &compute.semaphores.complete));

		// Create a compute capable device queue
		vkGetDeviceQueue(ctx->deviceHandle(), ctx->queueFamilyIndices.compute, 0, &queue);
	}

	VkMultiProgram::~VkMultiProgram()
	{

	}

	void VkMultiProgram::add(std::string name, std::shared_ptr<VkProgram> program)
	{
		assert(program != nullptr);
		mPrograms[name] = program;
		program->setVkCommandBuffer(commandBuffers);
	}

	void VkMultiProgram::begin()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		cmdBufInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

		VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffers, &cmdBufInfo));

		for (auto &pgm : mPrograms) {
			pgm.second->suspendInherentCmdBuffer(commandBuffers);
		}
		//vkCmdBindPipeline(commandBuffers, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	}

	void VkMultiProgram::update(bool sync)
	{
		vkResetFences(ctx->deviceHandle(), 1, &mFence);

		//		static bool firstDraw = true;
		VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		// FIXME find a better way to do this (without using fences, which is much slower)
		VkPipelineStageFlags computeWaitDstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		// 		if (!firstDraw) {
		// 			computeSubmitInfo.waitSemaphoreCount = 1;
		// 			computeSubmitInfo.pWaitSemaphores = &compute.semaphores.ready;
		// 			computeSubmitInfo.pWaitDstStageMask = &computeWaitDstStageMask;
		// 		}
		// 		else {
		// 			firstDraw = false;
		// 		}
		// 		computeSubmitInfo.signalSemaphoreCount = 1;
		// 		computeSubmitInfo.pSignalSemaphores = &compute.semaphores.complete;
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &commandBuffers;

		if (sync) {
			VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &computeSubmitInfo, mFence));
			VK_CHECK_RESULT(vkWaitForFences(ctx->deviceHandle(), 1, &mFence, VK_TRUE, UINT64_MAX));
		}
		else {
			VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));
		}
	}

	void VkMultiProgram::end()
	{
		for (auto &pgm : mPrograms) {
			pgm.second->restoreInherentCmdBuffer();
		}

		// release the storage buffers back to the graphics queue
		vkEndCommandBuffer(commandBuffers);
	}

	void VkMultiProgram::wait()
	{
		VK_CHECK_RESULT(vkWaitForFences(ctx->deviceHandle(), 1, &mFence, VK_TRUE, UINT64_MAX));
	}
}