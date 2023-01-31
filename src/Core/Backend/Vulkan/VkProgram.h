#pragma once
#include "VkSystem.h"
#include "VkContext.h"
#include "VkDeviceArray.h"
#include "VkDeviceArray2D.h"
#include "VkDeviceArray3D.h"
#include "VkUniform.h"
#include "VkConstant.h"
#include <memory>
#include <map>

namespace dyno {

	using uint = unsigned int;

	struct uint3
	{
		uint x, y, z;
	};

	struct dim3
	{
		uint x = 1;
		uint y = 1;
		uint z = 1;
	};

	static uint iDivUp(uint a, uint b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	// compute grid and thread block size for a given number of elements
	static dim3 vkDispatchSize(uint totalSize, uint blockSize)
	{
		dim3 blockDim;
		blockDim.x = iDivUp(totalSize, blockSize);
		return blockDim;
	}

	static dim3 vkDispatchSize2D(uint size_x, uint size_y, uint blockSize)
	{
		dim3 blockDims;
		blockDims.x = iDivUp(size_x, blockSize);
		blockDims.y = iDivUp(size_y, blockSize);
		blockDims.z = 1;

		return blockDims;
	}

	static dim3 vkDispatchSize3D(uint size_x, uint size_y, uint size_z, uint blockSize)
	{
		dim3 blockDims;
		blockDims.x = iDivUp(size_x, blockSize);
		blockDims.y = iDivUp(size_y, blockSize);
		blockDims.z = iDivUp(size_z, blockSize);

		return blockDims;
	}

	template<typename T>
	inline VkDeviceArray<T>* bufferPtr()
	{
		static VkDeviceArray<T> var;
		return &var;
	}

	template<typename T>
	inline VkDeviceArray2D<T>* buffer2DPtr()
	{
		static VkDeviceArray2D<T> var;
		return &var;
	}

	template<typename T>
	inline VkDeviceArray3D<T>* buffer3DPtr()
	{
		static VkDeviceArray3D<T> var;
		return &var;
	}

	template<typename T>
	inline VkUniform<T>* uniformPtr()
	{
		static VkUniform<T> var;
		return &var;
	}

	template<typename T>
	inline VkConstant<T>* constantPtr()
	{
		static VkConstant<T> var;
		return &var;
	}

#define BUFFER(T) bufferPtr<T>()
#define BUFFER2D(T) buffer2DPtr<T>()
#define BUFFER3D(T) buffer3DPtr<T>()
#define UNIFORM(T) uniformPtr<T>()
#define CONSTANT(T) constantPtr<T>()

	class VkProgram {

	public:
		template<typename... Args>
		VkProgram(Args... args);

		~VkProgram();


		void begin();

		void setupArgs(std::vector<VkVariable*>& vars, VkVariable* t)
		{
			vars.push_back(t);
		}

		template<typename... Args>
		void enqueue(dim3 groupSize, Args... args);

		template<typename... Args>
		void write(Args... args);

		void dispatch(dim3 groupSize);

		void end();

		void update(bool sync = false);

		void wait();

		template<typename... Args>
		void flush(dim3 groupSize, Args... args) {
			this->begin();
			this->enqueue(groupSize, args...);
			this->end();

			this->update(true);
			this->wait();
		}

		bool load(std::string fileName);
        void addMacro(std::string key, std::string value);

		void addGraphicsToComputeBarriers(VkCommandBuffer commandBuffer);
		void addComputeToComputeBarriers(VkCommandBuffer commandBuffer);
		void addComputeToGraphicsBarriers(VkCommandBuffer commandBuffer);

		// Resources for the compute part of the example
		struct {
			struct Semaphores {
				VkSemaphore ready{ 0L };
				VkSemaphore complete{ 0L };
			} semaphores;
		} compute;

		void setVkCommandBuffer(VkCommandBuffer cmdBuffer) { mCommandBuffers = cmdBuffer; }
		void bindPipeline();

		void suspendInherentCmdBuffer(VkCommandBuffer cmdBuffer);
		void restoreInherentCmdBuffer();

	protected:
		void pushArgument(VkVariable* arg);
		void pushConstant(VkVariable* arg);
		void pushDeviceBuffer(VkVariable* arg);
		void pushUniform(VkVariable* arg);

	private:
		VkPipelineShaderStageCreateInfo createComputeStage(std::string fileName);

		VkContext* ctx = nullptr;

		std::vector<VkVariable*> mAllArgs;
		std::vector<VkVariable*> mBufferArgs;
		std::vector<VkVariable*> mUniformArgs;
		std::vector<VkVariable*> mConstArgs;

		// Descriptor set pool
		VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
		VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
		VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;
		
		VkFence mFence;
		VkQueue queue = VK_NULL_HANDLE;
		VkCommandPool mCommandPool = VK_NULL_HANDLE;
		VkCommandBuffer mCommandBuffers = VK_NULL_HANDLE;

		VkCommandBuffer mCmdBufferCopy = VK_NULL_HANDLE;
		std::vector<VkShaderModule> shaderModules;
	};

	template<typename... Args>
	VkProgram::VkProgram(Args... args)
	{
		ctx = VkSystem::instance()->currentContext();

		// Create the command pool
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = ctx->queueFamilyIndices.compute;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(ctx->deviceHandle(), &cmdPoolInfo, nullptr, &mCommandPool));

		// Create a command buffer for compute operations
		VkCommandBufferAllocateInfo cmdBufAllocateInfo =
			vks::initializers::commandBufferAllocateInfo(mCommandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);

		VK_CHECK_RESULT(vkAllocateCommandBuffers(ctx->deviceHandle(), &cmdBufAllocateInfo, &mCommandBuffers));

		// Semaphores for graphics / compute synchronization
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(ctx->deviceHandle(), &semaphoreCreateInfo, nullptr, &compute.semaphores.ready));
		VK_CHECK_RESULT(vkCreateSemaphore(ctx->deviceHandle(), &semaphoreCreateInfo, nullptr, &compute.semaphores.complete));

		uint32_t nBuffer = 0;
		uint32_t nUniform = 0;
		std::initializer_list<VkVariable*> variables{args...};
		for (auto variable : variables)
		{
			switch (variable->type())
			{
			case VariableType::DeviceBuffer:
				this->pushDeviceBuffer(variable);
				nBuffer++;
				break;
			case VariableType::Uniform:
				this->pushUniform(variable);
				nUniform++;
				break;
			case VariableType::Constant:
				this->pushConstant(variable);
				break;
			default:
				break;
			}
		}

		//Create descriptor set layout
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
		for (size_t i = 0; i < mAllArgs.size(); i++)
		{
			if (mAllArgs[i]->type() == VariableType::DeviceBuffer || mAllArgs[i]->type() == VariableType::Uniform)
			{
				setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(VkVariable::descriptorType(mAllArgs[i]->type()), VK_SHADER_STAGE_COMPUTE_BIT, i));
			}
		}

		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(ctx->deviceHandle(), &descriptorLayout, nullptr, &descriptorSetLayout));


		std::vector<VkDescriptorPoolSize> poolSizes;
		
		if (nBuffer > 0)
			poolSizes.push_back(vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nBuffer));

		if (nUniform > 0)
			poolSizes.push_back(vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nUniform));

		// Create the descriptor pool
		VkDescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(poolSizes, poolSizes.size());
		VK_CHECK_RESULT(vkCreateDescriptorPool(ctx->deviceHandle(), &descriptorPoolInfo, nullptr, &descriptorPool));

		VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

		// Create two descriptor sets with input and output buffers switched
		VK_CHECK_RESULT(vkAllocateDescriptorSets(ctx->deviceHandle(), &allocInfo, &descriptorSet));

		// Create a compute capable device queue
		vkGetDeviceQueue(ctx->deviceHandle(), ctx->queueFamilyIndices.compute, 0, &queue);

		VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
		VK_CHECK_RESULT(vkCreateFence(ctx->deviceHandle(), &fenceInfo, nullptr, &mFence));
	}

	template<typename... Args>
	void VkProgram::enqueue(dim3 groupSize, Args... args)
	{
		std::initializer_list<VkVariable*> variables{args...};

		mAllArgs.clear();
		mBufferArgs.clear();
		mUniformArgs.clear();
		mConstArgs.clear();
		for (auto variable : variables)
		{
			switch (variable->type())
			{
			case VariableType::DeviceBuffer:
				this->pushDeviceBuffer(variable);
				break;
			case VariableType::Uniform:
				this->pushUniform(variable);
				break;
			case VariableType::Constant:
				this->pushConstant(variable);
				break;
			default:
				break;
			}
		}

#ifndef NDEBUG
		assert(mAllArgs.size() == variables.size());
		for (std::size_t i = 0; i < variables.size(); i++)
		{
			auto variable = *(variables.begin() + i);
			assert(mAllArgs[i]->type() == variable->type());
		}
#endif // !NDEBUG


		//Create descriptor set layout
		std::vector<VkWriteDescriptorSet> writeDescriptorSets;
		for (size_t i = 0; i < variables.size(); i++)
		{
			auto variable = *(variables.begin() + i);
			if ((mAllArgs[i]->type() == VariableType::DeviceBuffer && mAllArgs[i]->bufferSize() > 0) || mAllArgs[i]->type() == VariableType::Uniform) {
				writeDescriptorSets.push_back(
					vks::initializers::writeDescriptorSet(descriptorSet, VkVariable::descriptorType(variable->type()), i, &variable->getDescriptor()));
			}
		}

		vkUpdateDescriptorSets(ctx->deviceHandle(), static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
		vkCmdBindDescriptorSets(mCommandBuffers, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, 0);
		writeDescriptorSets.clear();
		
		vkCmdBindPipeline(mCommandBuffers, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

		uint32_t offset = 0;
		for (size_t i = 0; i < variables.size(); i++)
		{
			auto variable = *(variables.begin() + i);
			if (variable->type() == VariableType::Constant) {
				vkCmdPushConstants(mCommandBuffers, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, offset, variable->bufferSize(), variable->data());
				offset += variable->bufferSize();
			}
		}

		vkCmdDispatch(mCommandBuffers, groupSize.x, groupSize.y, groupSize.z);

		addComputeToComputeBarriers(mCommandBuffers);
	}

	template<typename... Args>
	void VkProgram::write(Args... args)
	{
		std::initializer_list<VkVariable*> variables{ args... };

		mAllArgs.clear();
		mBufferArgs.clear();
		mUniformArgs.clear();
		mConstArgs.clear();
		for (auto variable : variables)
		{
			switch (variable->type())
			{
			case VariableType::DeviceBuffer:
				this->pushDeviceBuffer(variable);
				break;
			case VariableType::Uniform:
				this->pushUniform(variable);
				break;
			case VariableType::Constant:
				this->pushConstant(variable);
				break;
			default:
				break;
			}
		}

#ifndef NDEBUG
		assert(mAllArgs.size() == variables.size());
		for (std::size_t i = 0; i < variables.size(); i++)
		{
			auto variable = *(variables.begin() + i);
			assert(mAllArgs[i]->type() == variable->type());
		}
#endif // !NDEBUG


		//Create descriptor set layout
		std::vector<VkWriteDescriptorSet> writeDescriptorSets;
		for (size_t i = 0; i < variables.size(); i++)
		{
			auto variable = *(variables.begin() + i);
			if ((mAllArgs[i]->type() == VariableType::DeviceBuffer && mAllArgs[i]->bufferSize() > 0) || mAllArgs[i]->type() == VariableType::Uniform) {
				writeDescriptorSets.push_back(
					vks::initializers::writeDescriptorSet(descriptorSet, VkVariable::descriptorType(variable->type()), i, &variable->getDescriptor()));
			}
		}

		vkUpdateDescriptorSets(ctx->deviceHandle(), static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
		writeDescriptorSets.clear();
	}



	class VkMultiProgram
	{
	public:
		VkMultiProgram();
		~VkMultiProgram();

		void add(std::string name, std::shared_ptr<VkProgram> program);

		inline std::shared_ptr<VkProgram> operator [] (std::string name)
		{
			return mPrograms[name];
		}

		void begin();
		void update(bool sync = false);
		void end();

		void wait();

		// Resources for the compute part of the example
		struct {
			struct Semaphores {
				VkSemaphore ready{ 0L };
				VkSemaphore complete{ 0L };
			} semaphores;
		} compute;

	private:
		std::map<std::string, std::shared_ptr<VkProgram>> mPrograms;

		VkContext* ctx = nullptr;

		VkFence mFence;
		VkQueue queue = VK_NULL_HANDLE;
		VkCommandPool commandPool = VK_NULL_HANDLE;
		VkCommandBuffer commandBuffers = VK_NULL_HANDLE;
	};
}