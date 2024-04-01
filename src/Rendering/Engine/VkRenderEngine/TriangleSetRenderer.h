#pragma once 
#include <array>

#include "FilePath.h"
#include "VkSystem.h"
#include "VulkanTools.h"
#include "VkContext.h"
#include "VkSystem.h"
#include "VkDeviceArray.h"
#include "VkUniform.h"
#include "VkVisualModule.h"

#include "Topology/TriangleSet.h"

using namespace dyno;

namespace dyno {

	struct Vertex
	{
		glm::vec4 pos;
		glm::vec4 uv;
		glm::vec4 normal;
	};

	//TODO: implement an easy to use graphics pipeline that can be adapted into the framework
	class TriangleSetRenderer : public dyno::VkVisualModule
	{
	public:
		TriangleSetRenderer();

		~TriangleSetRenderer();

		DEF_INSTANCE_IN(TriangleSet3f, Topology, "");

		void prepare(VkRenderPass renderPass) override
		{
			this->updateGraphicsUBO();

			this->setupDescriptorPool();
			this->setupLayoutsAndDescriptors();

			this->preparePipelines(renderPass);
		}

		void setupDescriptorPool();

		void setupLayoutsAndDescriptors();

		void preparePipelines(VkRenderPass renderPass);

		void addGraphicsToComputeBarriers(VkCommandBuffer commandBuffer);

		void addComputeToGraphicsBarriers(VkCommandBuffer commandBuffer);

		void buildCommandBuffers(VkCommandBuffer drawCmdBuffer) override;

		void updateGraphicsUBO()
		{
			mUniform.setValue(ubo);
		}

		void viewChanged(const glm::mat4& perspective, const glm::mat4& view) override
		{
			ubo.projection = perspective;
			ubo.view = view;
			mUniform.setValue(ubo);
		}

	private:
		std::shared_ptr<VkProgram> program;

		VkConstant<uint32_t> particleNumber;

	protected:
		bool initializeImpl() override;
		void updateGraphicsContext() override;

		FilePath getShadersPath() const;

		VkPipelineShaderStageCreateInfo loadShader(std::string fileName, VkShaderStageFlagBits stage);

	private:
		std::string shaderDir = "glsl";

		VkContext* ctx;

		VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

		bool specializedComputeQueue = false;

		// Resources for the rendering pipeline
		VkDescriptorSetLayout descriptorSetLayout;
		VkDescriptorSet descriptorSet;
		VkPipelineLayout pipelineLayout;
		VkPipeline pipeline;
		std::vector<VkShaderModule> shaderModules;

	public:
		struct GraphicsUBO {
			glm::mat4 projection;
			glm::mat4 view;
			glm::vec4 lightPos = glm::vec4(-2.0f, 4.0f, -2.0f, 1.0f);
		} ubo;

		VkDeviceArray<Vertex> mVertex;
		VkDeviceArray<uint32_t> mIndex;

		VkUniform<GraphicsUBO> mUniform;
	};
}

