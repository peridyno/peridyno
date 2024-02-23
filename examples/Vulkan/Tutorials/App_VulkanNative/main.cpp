/**
 * Copyright 2023 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "vulkan/vulkan.h"

#include <string>
#include <iostream>
#include <vector>
#include <assert.h>

#include "VkContext.h"
#include "VkProgram.h"
#include "Array/Array.h"
#include "PlatformConfig.h"

using namespace dyno;

/**
 * This example is used to demonstrate how to use native APIs in Vulkan to add two arrays in parallel
 */

int main(int, char**)
{
	/************************************************************************/
	/* Create instance                                                      */
	/************************************************************************/
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "Vulkan";
	appInfo.pEngineName = "Vulkan";
	appInfo.apiVersion = VK_API_VERSION_1_2;

	std::vector<const char*> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };

#if defined(_WIN32)
	instanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
	instanceExtensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(_DIRECT2DISPLAY)
	instanceExtensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
	instanceExtensions.push_back(VK_EXT_DIRECTFB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
	instanceExtensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
	instanceExtensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
	instanceExtensions.push_back(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
	instanceExtensions.push_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#endif

	VkInstanceCreateInfo instanceCreateInfo = {};
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pNext = NULL;
	instanceCreateInfo.pApplicationInfo = &appInfo;

	if (instanceExtensions.size() > 0) {
		instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
		instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
	}

	VkInstance vkInstance;

	VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &vkInstance);

	/************************************************************************/
	/* Pick a physical device                                               */
	/************************************************************************/
	uint32_t gpuCount = 0;
	vkEnumeratePhysicalDevices(vkInstance, &gpuCount, nullptr);
	assert(gpuCount > 0);

	std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
	VkResult err = vkEnumeratePhysicalDevices(vkInstance, &gpuCount, physicalDevices.data());
	if (err) {
		return false;
	}

	uint32_t selectedDevice = 0;

	VkPhysicalDevice physicalDevice = physicalDevices[selectedDevice];

	/************************************************************************/
	/* Create a logical device                                              */
	/************************************************************************/
	VkPhysicalDeviceFeatures enabledFeatures{};
	std::vector<const char*> enabledDeviceExtensions;
	void* deviceCreatepNextChain = nullptr;

	dyno::VkContext* ctx = new dyno::VkContext(physicalDevice);
	VkResult res = ctx->createLogicalDevice(enabledFeatures, enabledDeviceExtensions, deviceCreatepNextChain);
	if (res != VK_SUCCESS) {
		return false;
	}


	/************************************************************************/
	/* Allocate memory and buffers                                          */
	/************************************************************************/
	uint num = 100;

	uint32_t bufferSize = num * sizeof(float);
	std::shared_ptr<vks::Buffer> bufferA = std::make_shared<vks::Buffer>(ctx->deviceHandle());
	std::shared_ptr<vks::Buffer> bufferB = std::make_shared<vks::Buffer>(ctx->deviceHandle());
	std::shared_ptr<vks::Buffer> bufferC = std::make_shared<vks::Buffer>(ctx->deviceHandle());
	ctx->createBuffer(
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		bufferA,
		bufferSize);

	ctx->createBuffer(
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		bufferB,
		bufferSize);

	ctx->createBuffer(
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		bufferC,
		bufferSize);

	CArray<float> hA(num);
	CArray<float> hB(num);
	CArray<float> hC(num);

	for (int i = 0; i < num; i++)
	{
		hA[i] = float(i);
		hB[i] = float(i);
	}

	std::shared_ptr<vks::Buffer> stageA = std::make_shared<vks::Buffer>(ctx->deviceHandle());
	std::shared_ptr<vks::Buffer> stageB = std::make_shared<vks::Buffer>(ctx->deviceHandle());
	std::shared_ptr<vks::Buffer> stageC = std::make_shared<vks::Buffer>(ctx->deviceHandle());
	ctx->createBuffer(
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		stageA,
		bufferSize,
		hA.begin());

	ctx->createBuffer(
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		stageB,
		bufferSize,
		hB.begin());

	ctx->createBuffer(
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		stageC,
		bufferSize,
		nullptr);

	// Copy from staging buffer
	VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
	VkBufferCopy copyRegion = {};
	copyRegion.size = bufferSize;
	vkCmdCopyBuffer(copyCmd, stageA->buffer, bufferA->buffer, 1, &copyRegion);
	vkCmdCopyBuffer(copyCmd, stageB->buffer, bufferB->buffer, 1, &copyRegion);
	ctx->flushCommandBuffer(copyCmd, ctx->transferQueueHandle(), true);

	/************************************************************************/
	/* Create descriptor sets and compute pipelines                         */
	/************************************************************************/
	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
	setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(VkVariable::descriptorType(VariableType::DeviceBuffer), VK_SHADER_STAGE_COMPUTE_BIT, 0));
	setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(VkVariable::descriptorType(VariableType::DeviceBuffer), VK_SHADER_STAGE_COMPUTE_BIT, 1));
	setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(VkVariable::descriptorType(VariableType::DeviceBuffer), VK_SHADER_STAGE_COMPUTE_BIT, 2));

	VkDescriptorSetLayoutCreateInfo descriptorLayout =
		vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);

	VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
	vkCreateDescriptorSetLayout(ctx->deviceHandle(), &descriptorLayout, nullptr, &descriptorSetLayout);

	std::vector<VkDescriptorPoolSize> poolSizes;
	poolSizes.push_back(vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3));

	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

	// Create the descriptor pool
	VkDescriptorPoolCreateInfo descriptorPoolInfo =
		vks::initializers::descriptorPoolCreateInfo(poolSizes, poolSizes.size());
	vkCreateDescriptorPool(ctx->deviceHandle(), &descriptorPoolInfo, nullptr, &descriptorPool);

	VkDescriptorSetAllocateInfo allocInfo =
		vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

	// Create two descriptor sets with input and output buffers switched
	VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
	VK_CHECK_RESULT(vkAllocateDescriptorSets(ctx->deviceHandle(), &allocInfo, &descriptorSet));

	// Create a compute capable device queue
	VkQueue queue = VK_NULL_HANDLE;
	vkGetDeviceQueue(ctx->deviceHandle(), ctx->queueFamilyIndices.compute, 0, &queue);

	VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
	VkFence mFence;
	vkCreateFence(ctx->deviceHandle(), &fenceInfo, nullptr, &mFence);

	//Create pipeline layout
	std::vector<VkPushConstantRange> pushConstantRanges;
	pushConstantRanges.push_back(vks::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(uint), 0));

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
		vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
	pipelineLayoutCreateInfo.pushConstantRangeCount = pushConstantRanges.size();
	pipelineLayoutCreateInfo.pPushConstantRanges = pushConstantRanges.data();

	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	vkCreatePipelineLayout(ctx->deviceHandle(), &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

	// Create pipeline
	VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(pipelineLayout, 0);
	std::string fileName = getAssetPath()  / "shaders/glsl/tutorials/VecAdd.comp.spv";
	VkPipelineShaderStageCreateInfo shaderStage = {};
	shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStage.module = vks::tools::loadShaderModule(fileName, ctx->deviceHandle());
	shaderStage.pName = "main";
	assert(shaderStage.module != VK_NULL_HANDLE);
	computePipelineCreateInfo.stage = shaderStage;
	VkPipeline pipeline = VK_NULL_HANDLE;
	vkCreateComputePipelines(ctx->deviceHandle(), ctx->pipelineCacheHandle(), 1, &computePipelineCreateInfo, nullptr, &pipeline);

	/************************************************************************/
	/* Create Command Buffers                                               */
	/************************************************************************/
	VkCommandPool mCommandPool = VK_NULL_HANDLE;
	// Create the command pool
	VkCommandPoolCreateInfo cmdPoolInfo = {};
	cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmdPoolInfo.queueFamilyIndex = ctx->queueFamilyIndices.compute;
	cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	vkCreateCommandPool(ctx->deviceHandle(), &cmdPoolInfo, nullptr, &mCommandPool);

	// Create a command buffer for compute operations
	VkCommandBufferAllocateInfo cmdBufAllocateInfo =
		vks::initializers::commandBufferAllocateInfo(mCommandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);

	VkCommandBuffer mCommandBuffers = VK_NULL_HANDLE;
	vkAllocateCommandBuffers(ctx->deviceHandle(), &cmdBufAllocateInfo, &mCommandBuffers);

	VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
	cmdBufInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

	vkBeginCommandBuffer(mCommandBuffers, &cmdBufInfo);

	//Create descriptor set layout
	std::vector<VkWriteDescriptorSet> writeDescriptorSets;
	writeDescriptorSets.push_back(
		vks::initializers::writeDescriptorSet(descriptorSet, VkVariable::descriptorType(VariableType::DeviceBuffer), 0, &bufferA->descriptor));
	writeDescriptorSets.push_back(
		vks::initializers::writeDescriptorSet(descriptorSet, VkVariable::descriptorType(VariableType::DeviceBuffer), 1, &bufferB->descriptor));
	writeDescriptorSets.push_back(
		vks::initializers::writeDescriptorSet(descriptorSet, VkVariable::descriptorType(VariableType::DeviceBuffer), 2, &bufferC->descriptor));

	vkUpdateDescriptorSets(ctx->deviceHandle(), static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
	vkCmdBindDescriptorSets(mCommandBuffers, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, 0);
	writeDescriptorSets.clear();

	vkCmdBindPipeline(mCommandBuffers, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

	uint32_t offset = 0;
	vkCmdPushConstants(mCommandBuffers, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint), (void*)&num);

	dim3 groupSize = vkDispatchSize(num, 128);
	vkCmdDispatch(mCommandBuffers, groupSize.x, groupSize.y, groupSize.z);

	VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
	bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	bufferBarrier.srcQueueFamilyIndex = ctx->queueFamilyIndices.compute;
	bufferBarrier.dstQueueFamilyIndex = ctx->queueFamilyIndices.compute;
	bufferBarrier.size = VK_WHOLE_SIZE;
	std::vector<VkBufferMemoryBarrier> bufferBarriers;
	bufferBarrier.buffer = bufferA->buffer;
	bufferBarriers.push_back(bufferBarrier);
	bufferBarrier.buffer = bufferB->buffer;
	bufferBarriers.push_back(bufferBarrier);
	bufferBarrier.buffer = bufferC->buffer;
	bufferBarriers.push_back(bufferBarrier);

	vkCmdPipelineBarrier(
		mCommandBuffers,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_FLAGS_NONE,
		0, nullptr,
		static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),
		0, nullptr);

	vkEndCommandBuffer(mCommandBuffers);

	/************************************************************************/
	/* Dispatch		                                                        */
	/************************************************************************/

	vkResetFences(ctx->deviceHandle(), 1, &mFence);

	static bool firstDraw = true;
	VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
	// FIXME find a better way to do this (without using fences, which is much slower)
	VkPipelineStageFlags computeWaitDstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	computeSubmitInfo.commandBufferCount = 1;
	computeSubmitInfo.pCommandBuffers = &mCommandBuffers;

	vkQueueSubmit(queue, 1, &computeSubmitInfo, mFence);
	vkWaitForFences(ctx->deviceHandle(), 1, &mFence, VK_TRUE, UINT64_MAX);


	/************************************************************************/
	/* Read results back                                                    */
	/************************************************************************/
	VkCommandBuffer copyBackCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
	VkBufferCopy copyBackRegion = {};
	copyBackRegion.size = num * sizeof(float);
	vkCmdCopyBuffer(copyBackCmd, bufferC->buffer, stageC->buffer, 1, &copyBackRegion);
	ctx->flushCommandBuffer(copyBackCmd, ctx->transferQueueHandle(), true);

	std::vector<float> ret(num);

	stageC->map();
	memcpy(ret.data(), stageC->mapped, sizeof(float)* num);

	for (int i = 0; i < num; i++)
	{
		printf("%f \n", ret[i]);
	}

	ret.clear();

	return 0;
}
