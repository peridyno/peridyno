#include "TriangleSetRenderer.h"
#include "VkTransfer.h"
#include "Node.h"

namespace px 
{
	TriangleSetRenderer::TriangleSetRenderer()
		: VkVisualModule()
	{
		ctx = VkSystem::instance()->currentContext();

		// Check whether the compute queue family is distinct from the graphics queue family
		specializedComputeQueue = ctx->queueFamilyIndices.graphics != ctx->queueFamilyIndices.compute;
	};

	TriangleSetRenderer::~TriangleSetRenderer()
	{
		// Graphics
		for (auto& shderModule : shaderModules) {
			vkDestroyShaderModule(ctx->deviceHandle(), shderModule, nullptr);
		}
		vkDestroyPipeline(ctx->deviceHandle(), pipeline, nullptr);
		vkDestroyPipelineLayout(ctx->deviceHandle(), pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(ctx->deviceHandle(), descriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(ctx->deviceHandle(), descriptorPool, nullptr);
	};

	void TriangleSetRenderer::setupDescriptorPool()
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

	void TriangleSetRenderer::setupLayoutsAndDescriptors()
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

	void TriangleSetRenderer::preparePipelines(VkRenderPass renderPass)
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

		shaderStages[0] = loadShader(getShadersPath() + "graphics/cloth.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "graphics/cloth.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);

		// Input attributes

		// Binding description
		std::vector<VkVertexInputBindingDescription> inputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(px::Vertex), VK_VERTEX_INPUT_RATE_VERTEX)
		};

		// Attribute descriptions
		std::vector<VkVertexInputAttributeDescription> inputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)),
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)),
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal))
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
	}

	void TriangleSetRenderer::addGraphicsToComputeBarriers(VkCommandBuffer commandBuffer)
	{
		if (specializedComputeQueue) {
			VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
			bufferBarrier.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			bufferBarrier.srcQueueFamilyIndex = ctx->queueFamilyIndices.graphics;
			bufferBarrier.dstQueueFamilyIndex = ctx->queueFamilyIndices.compute;
			bufferBarrier.size = VK_WHOLE_SIZE;

			std::vector<VkBufferMemoryBarrier> bufferBarriers;
			bufferBarrier.buffer = mVertex.bufferHandle();
			bufferBarriers.push_back(bufferBarrier);
			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),
				0, nullptr);
		}
	}

	void TriangleSetRenderer::addComputeToGraphicsBarriers(VkCommandBuffer commandBuffer)
	{
		if (specializedComputeQueue) {
			VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
			bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			bufferBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
			bufferBarrier.srcQueueFamilyIndex = ctx->queueFamilyIndices.compute;
			bufferBarrier.dstQueueFamilyIndex = ctx->queueFamilyIndices.graphics;
			bufferBarrier.size = VK_WHOLE_SIZE;
			std::vector<VkBufferMemoryBarrier> bufferBarriers;
			bufferBarrier.buffer = mVertex.bufferHandle();
			bufferBarriers.push_back(bufferBarrier);
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

	void TriangleSetRenderer::buildCommandBuffers(VkCommandBuffer drawCmdBuffer)
	{
		// Acquire storage buffers from compute queue
		addComputeToGraphicsBarriers(drawCmdBuffer);

		VkDeviceSize offsets[1] = { 0 };

		// Render cloth
		if (mVertex.size() > 0)
		{
			VkBuffer dataBuffer = mVertex.bufferHandle();

			vkCmdBindPipeline(drawCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
			vkCmdBindDescriptorSets(drawCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
			vkCmdBindIndexBuffer(drawCmdBuffer, mIndex.bufferHandle(), 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindVertexBuffers(drawCmdBuffer, 0, 1, &dataBuffer, offsets);
			vkCmdDrawIndexed(drawCmdBuffer, mIndex.size(), 1, 0, 0, 0);
		}

		// release the storage buffers to the compute queue
		addGraphicsToComputeBarriers(drawCmdBuffer);
	}

	std::string TriangleSetRenderer::getShadersPath() const
	{
		return getAssetPath() + "shaders/" + shaderDir + "/";
	}

	VkPipelineShaderStageCreateInfo TriangleSetRenderer::loadShader(std::string fileName, VkShaderStageFlagBits stage)
	{
		VkPipelineShaderStageCreateInfo shaderStage = {};
		shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.stage = stage;
		shaderStage.module = vks::tools::loadShaderModule(fileName, ctx->deviceHandle());
		shaderStage.pName = "main";
		assert(shaderStage.module != VK_NULL_HANDLE);
		return shaderStage;
	}

	bool TriangleSetRenderer::initializeImpl()
	{
		auto triSet = std::dynamic_pointer_cast<TriangleSet>(this->inTopology()->getDataPtr());
		if (triSet == nullptr)
			return false;

		this->mIndex.resize(triSet->mIndex.size());
		vkTransfer(this->mIndex, triSet->mIndex);

		this->mVertex.resize(3 * triSet->mTriangleIndex.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		this->mIndex.resize(3 * triSet->mTriangleIndex.size());

		program = std::make_shared<VkProgram>(BUFFER(Vertex), BUFFER(uint32_t), BUFFER(Vec3f), BUFFER(uint32_t), CONSTANT(uint32_t));
		program->load(getAssetPath() + "shaders/glsl/graphics/SetupVertexFromPoints.comp.spv");

		particleNumber.setValue(triSet->mTriangleIndex.size());
		dim3 groupSize = vkDispatchSize(triSet->mTriangleIndex.size(), 32);
		program->begin();
		program->enqueue(groupSize, &this->mVertex, &this->mIndex, &triSet->mPoints, &triSet->mIndex, &this->particleNumber);
		program->end();

		program->update();
		
		return true;
	}

	void TriangleSetRenderer::updateGraphicsContext()
	{
		auto triSet = std::dynamic_pointer_cast<TriangleSet>(this->inTopology()->getDataPtr());
		assert(triSet != nullptr);

		program->update();
	}
}