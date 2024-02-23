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
#include "Primitive/Primitive3D.h"

namespace dyno {

	struct Vertex
	{
		glm::vec4 pos;
		glm::vec3 normal;
		glm::vec2 uv;
	};

	//TODO: implement an easy to use graphics pipeline that can be adapted into the framework
	class VkGraphicsPipeline : public dyno::VkVisualModule
	{
	public:
		VkGraphicsPipeline();

		~VkGraphicsPipeline();

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

// 		void addGraphicsToComputeBarriers(VkCommandBuffer commandBuffer);
// 
// 		void addComputeToGraphicsBarriers(VkCommandBuffer commandBuffer);

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

	protected:
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
		VkPipeline spherePipeline;
		VkPipeline capsulePipeline;
		VkPipeline trianglePipeline;
		std::vector<VkShaderModule> shaderModules;

	public:
		struct GraphicsUBO {
			glm::mat4 projection;
			glm::mat4 view;
			glm::vec4 lightPos = glm::vec4(-2.0f, 4.0f, -2.0f, 1.0f);
		} ubo;

// 		VkDeviceArray<Vertex> mVertex;
// 		VkDeviceArray<uint32_t> mIndex;

		VkDeviceArray<Vertex> mCubeVertex;
		VkDeviceArray<uint32_t> mCubeIndex;
		VkDeviceArray<px::Box> mCubeInstanceData;

		VkDeviceArray<Vertex> mSphereVertex;
		VkDeviceArray<uint32_t> mSphereIndex;
		VkDeviceArray<px::Sphere> mSphereInstanceData;

		VkDeviceArray<Vertex> mCapsuleVertex;
		VkDeviceArray<uint32_t> mCapsuleIndex;
		VkDeviceArray<px::Capsule> mCapsuleInstanceData;

		VkUniform<GraphicsUBO> mUniform;
	};
}

