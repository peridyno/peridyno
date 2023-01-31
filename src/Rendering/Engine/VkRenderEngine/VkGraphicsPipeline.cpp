#include "VkGraphicsPipeline.h"
#include "VkTransfer.h"
#include "Primitive/Primitive3D.h"

namespace dyno
{
	VkGraphicsPipeline::VkGraphicsPipeline()
		: VkVisualModule()
	{
		ctx = VkSystem::instance()->currentContext();

		// Check whether the compute queue family is distinct from the graphics queue family
		specializedComputeQueue = ctx->queueFamilyIndices.graphics != ctx->queueFamilyIndices.compute;
	};

	VkGraphicsPipeline::~VkGraphicsPipeline()
	{
		// Graphics
		for (auto& shderModule : shaderModules) {
			vkDestroyShaderModule(ctx->deviceHandle(), shderModule, nullptr);
		}
		vkDestroyPipeline(ctx->deviceHandle(), pipeline, nullptr);
		vkDestroyPipeline(ctx->deviceHandle(), spherePipeline, nullptr);
		vkDestroyPipeline(ctx->deviceHandle(), capsulePipeline, nullptr);
		vkDestroyPipelineLayout(ctx->deviceHandle(), pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(ctx->deviceHandle(), descriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(ctx->deviceHandle(), descriptorPool, nullptr);
	};

	void VkGraphicsPipeline::setupDescriptorPool()
	{
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2)
		};

		VkDescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(poolSizes, 3);

		VK_CHECK_RESULT(vkCreateDescriptorPool(ctx->deviceHandle(), &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	void VkGraphicsPipeline::setupLayoutsAndDescriptors()
	{
		// Set layout
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0)
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(ctx->deviceHandle(), &descriptorLayout, nullptr, &descriptorSetLayout));

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(ctx->deviceHandle(), &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		// Set
		VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(ctx->deviceHandle(), &allocInfo, &descriptorSet));

		std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &mUniform.getDescriptor())
		};
		vkUpdateDescriptorSets(ctx->deviceHandle(), static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
	}

	void VkGraphicsPipeline::preparePipelines(VkRenderPass renderPass)
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);

		VkPipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);

		VkPipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);

		VkPipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

		VkPipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);

		VkPipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

		VkPipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);

		std::vector<VkDynamicState> dynamicStateEnables = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables, 0);

		// Rendering pipeline
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		shaderStages[0] = loadShader(getShadersPath() + "graphics/render.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "graphics/render.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);

		// Input attributes

		// Binding description
		std::vector<VkVertexInputBindingDescription> inputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
			vks::initializers::vertexInputBindingDescription(1, sizeof(px::Box), VK_VERTEX_INPUT_RATE_INSTANCE)

		};

		// Attribute descriptions
		std::vector<VkVertexInputAttributeDescription> inputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)),
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)),
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)),

			vks::initializers::vertexInputAttributeDescription(1, 3, VK_FORMAT_R32G32B32_SFLOAT, offsetof(px::Box, center)),
			vks::initializers::vertexInputAttributeDescription(1, 4, VK_FORMAT_R32G32B32_SFLOAT, offsetof(px::Box, halfLength)),
			vks::initializers::vertexInputAttributeDescription(1, 5, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(px::Box, rot))
		};

		// Assign to vertex buffer
		VkPipelineVertexInputStateCreateInfo inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(inputBindings.size());
		inputState.pVertexBindingDescriptions = inputBindings.data();
		inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
		inputState.pVertexAttributeDescriptions = inputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &inputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.renderPass = renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(ctx->deviceHandle(), ctx->pipelineCacheHandle(), 1, &pipelineCreateInfo, nullptr, &pipeline));
	
		// create sphere pipeline
		shaderStages[0] = loadShader(getShadersPath() + "graphics/renderSphere.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);

		inputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
			vks::initializers::vertexInputBindingDescription(1, sizeof(px::Sphere), VK_VERTEX_INPUT_RATE_INSTANCE)

		};

		inputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)),
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)),
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)),

			vks::initializers::vertexInputAttributeDescription(1, 3, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(px::Sphere, rot)),
			vks::initializers::vertexInputAttributeDescription(1, 4, VK_FORMAT_R32G32B32_SFLOAT, offsetof(px::Sphere, center)),
			vks::initializers::vertexInputAttributeDescription(1, 5, VK_FORMAT_R32_SFLOAT, offsetof(px::Sphere, radius))
		};
		inputState.pVertexBindingDescriptions = inputBindings.data();
		inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
		inputState.pVertexAttributeDescriptions = inputAttributes.data();
		pipelineCreateInfo.pVertexInputState = &inputState;
		pipelineCreateInfo.pStages = shaderStages.data();
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(ctx->deviceHandle(), ctx->pipelineCacheHandle(), 1, &pipelineCreateInfo, nullptr, &spherePipeline));

		// create capsule pipeline
		shaderStages[0] = loadShader(getShadersPath() + "graphics/renderCapsule.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);

		inputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
			vks::initializers::vertexInputBindingDescription(1, sizeof(px::Capsule), VK_VERTEX_INPUT_RATE_INSTANCE)

		};

		inputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)),
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)),
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)),

			vks::initializers::vertexInputAttributeDescription(1, 3, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(px::Capsule, rot)),
			vks::initializers::vertexInputAttributeDescription(1, 4, VK_FORMAT_R32G32B32_SFLOAT, offsetof(px::Capsule, center)),
			vks::initializers::vertexInputAttributeDescription(1, 5, VK_FORMAT_R32_SFLOAT, offsetof(px::Capsule, halfLength)),
			vks::initializers::vertexInputAttributeDescription(1, 6, VK_FORMAT_R32_SFLOAT, offsetof(px::Capsule, radius))
		};
		inputState.pVertexBindingDescriptions = inputBindings.data();
		inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
		inputState.pVertexAttributeDescriptions = inputAttributes.data();
		pipelineCreateInfo.pVertexInputState = &inputState;
		pipelineCreateInfo.pStages = shaderStages.data();
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(ctx->deviceHandle(), ctx->pipelineCacheHandle(), 1, &pipelineCreateInfo, nullptr, &capsulePipeline));
	}

// 	void VkGraphicsPipeline::addGraphicsToComputeBarriers(VkCommandBuffer commandBuffer)
// 	{
// 		if (specializedComputeQueue) {
// 			VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
// 			bufferBarrier.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
// 			bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
// 			bufferBarrier.srcQueueFamilyIndex = ctx->queueFamilyIndices.graphics;
// 			bufferBarrier.dstQueueFamilyIndex = ctx->queueFamilyIndices.compute;
// 			bufferBarrier.size = VK_WHOLE_SIZE;
// 
// 			std::vector<VkBufferMemoryBarrier> bufferBarriers;
// 			bufferBarrier.buffer = mVertex.bufferHandle();
// 			bufferBarriers.push_back(bufferBarrier);
// 			vkCmdPipelineBarrier(commandBuffer,
// 				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
// 				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
// 				VK_FLAGS_NONE,
// 				0, nullptr,
// 				static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),
// 				0, nullptr);
// 		}
// 	}
// 
// 	void VkGraphicsPipeline::addComputeToGraphicsBarriers(VkCommandBuffer commandBuffer)
// 	{
// 		if (specializedComputeQueue) {
// 			VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
// 			bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
// 			bufferBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
// 			bufferBarrier.srcQueueFamilyIndex = ctx->queueFamilyIndices.compute;
// 			bufferBarrier.dstQueueFamilyIndex = ctx->queueFamilyIndices.graphics;
// 			bufferBarrier.size = VK_WHOLE_SIZE;
// 			std::vector<VkBufferMemoryBarrier> bufferBarriers;
// 			bufferBarrier.buffer = mVertex.bufferHandle();
// 			bufferBarriers.push_back(bufferBarrier);
// 			vkCmdPipelineBarrier(
// 				commandBuffer,
// 				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
// 				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
// 				VK_FLAGS_NONE,
// 				0, nullptr,
// 				static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),
// 				0, nullptr);
// 		}
// 	}

	void VkGraphicsPipeline::buildCommandBuffers(VkCommandBuffer drawCmdBuffer)
	{
		VkDeviceSize offsets[1] = { 0 };
		VkBuffer dataBuffer = mCubeVertex.bufferHandle();
		VkBuffer instanceBuffer = mCubeInstanceData.bufferHandle();
		int cubeCount = mCubeInstanceData.size();
		// draw cube
		if (cubeCount > 0) {
			vkCmdBindPipeline(drawCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
			vkCmdBindDescriptorSets(drawCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
			vkCmdBindIndexBuffer(drawCmdBuffer, mCubeIndex.bufferHandle(), 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindVertexBuffers(drawCmdBuffer, 0, 1, &dataBuffer, offsets);
			vkCmdBindVertexBuffers(drawCmdBuffer, 1, 1, &instanceBuffer, offsets);
			vkCmdDrawIndexed(drawCmdBuffer, mCubeIndex.size(), cubeCount, 0, 0, 0);
		}

		// draw sphere
		int sphereCount = mSphereInstanceData.size();
		if (sphereCount > 0) {
			dataBuffer = mSphereVertex.bufferHandle();
			instanceBuffer = mSphereInstanceData.bufferHandle();

			vkCmdBindPipeline(drawCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, spherePipeline);
			vkCmdBindDescriptorSets(drawCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
			vkCmdBindIndexBuffer(drawCmdBuffer, mSphereIndex.bufferHandle(), 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindVertexBuffers(drawCmdBuffer, 0, 1, &dataBuffer, offsets);
			vkCmdBindVertexBuffers(drawCmdBuffer, 1, 1, &instanceBuffer, offsets);
			vkCmdDrawIndexed(drawCmdBuffer, mSphereIndex.size(), sphereCount, 0, 0, 0);
		}

		// draw Capsule
		int capsuleCount = mCapsuleInstanceData.size();
		if (capsuleCount > 0) {
			dataBuffer = mCapsuleVertex.bufferHandle();
			instanceBuffer = mCapsuleInstanceData.bufferHandle();

			vkCmdBindPipeline(drawCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, capsulePipeline);
			vkCmdBindDescriptorSets(drawCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
			vkCmdBindIndexBuffer(drawCmdBuffer, mCapsuleIndex.bufferHandle(), 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindVertexBuffers(drawCmdBuffer, 0, 1, &dataBuffer, offsets);
			vkCmdBindVertexBuffers(drawCmdBuffer, 1, 1, &instanceBuffer, offsets);
			vkCmdDrawIndexed(drawCmdBuffer, mCapsuleIndex.size(), capsuleCount, 0, 0, 0);
		}
	}

	std::string VkGraphicsPipeline::getShadersPath() const
	{
		return getAssetPath() + "shaders/" + shaderDir + "/";
	}

	VkPipelineShaderStageCreateInfo VkGraphicsPipeline::loadShader(std::string fileName, VkShaderStageFlagBits stage)
	{
		VkPipelineShaderStageCreateInfo shaderStage = {};
		shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.stage = stage;
		shaderStage.module = vks::tools::loadShaderModule(fileName, ctx->deviceHandle());
		shaderStage.pName = "main";
		assert(shaderStage.module != VK_NULL_HANDLE);
		shaderModules.push_back(shaderStage.module);
		return shaderStage;
	}

}