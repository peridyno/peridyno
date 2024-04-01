#pragma once

#include <memory>
#include <map>
#include <filesystem>
#include <array>
#include <cstddef>
#include <typeinfo>
#include <optional>
#include <type_traits>

#include "VkSystem.h"
#include "VkContext.h"
#include "VkDeviceArray.h"
#include "VkDeviceArray2D.h"
#include "VkDeviceArray3D.h"
#include "VkUniform.h"
#include "VkConstant.h"
#include "VkCompContext.h"


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

	static dim3 vkDispatchSize(const VkConstant<uint>& totalSize, uint blockSize)
	{
		dim3 blockDim;
		blockDim.x = iDivUp(totalSize.getValue(), blockSize);
		return blockDim;
	}

	static dim3 vkDispatchSize2D(const VkConstant<uint>& size_x, const VkConstant<uint>& size_y, uint blockSize)
	{
		dim3 blockDims;
		blockDims.x = iDivUp(size_x.getValue(), blockSize);
		blockDims.y = iDivUp(size_y.getValue(), blockSize);
		blockDims.z = 1;

		return blockDims;
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

	struct VkArgInfo {
		template<typename T, typename VT>
		static VkArgInfo info(VariableType type) {
			return VkArgInfo {type, &typeid(T), sizeof(VT), alignof(VT), std::is_scalar_v<VT>};
		}

		VariableType var_type;
		const std::type_info* type; // maybe template func addr?
		std::size_t var_size;
		std::size_t var_align;
		bool is_scalar;
	};

#define VKARGINFO(T, VT, VART) dyno::VkArgInfo::info<T, VT>(dyno::VariableType::VART)
#define BUFFER(T) VKARGINFO(VkDeviceArray<T>, T, DeviceBuffer)
#define BUFFER2D(T) VKARGINFO(VkDeviceArray2D<T>, T, DeviceBuffer)
#define BUFFER3D(T) VKARGINFO(VkDeviceArray3D<T>, T, DeviceBuffer)
#define UNIFORM(T) VKARGINFO(VkUniform<T>, T, Uniform)
#define CONSTANT(T) VKARGINFO(VkConstant<T>, T, Constant)

	class VkProgram {

	public:
		static constexpr int MaxDiscriptorSetNum {128};

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

		// void dispatch(dim3 groupSize);

		void end();

		auto update(bool sync = false) -> std::optional<VkFence>;

		void wait(VkFence);

		template<typename... Args>
		void submit(dim3 groupSize, Args... args) {
			auto& comp = VkCompContext::current();
			{
				this->begin();
				this->enqueue(groupSize, args...);
				this->end();

				this->update(false);
			}
		}

		template<typename... Args>
		void flush(dim3 groupSize, Args... args) {
			auto& comp = VkCompContext::current();
			comp.push();
			{
				this->begin();
				this->enqueue(groupSize, args...);
				this->end();

				auto fence = this->update(true);
				this->wait(fence.value());
			}
			comp.pop();
		}

		bool load(std::filesystem::path fileName);
        void addMacro(std::string key, std::string value);

		void addGraphicsToComputeBarriers(VkCommandBuffer commandBuffer, std::vector<VkVariable*>);
		void addComputeToComputeBarriers(VkCommandBuffer commandBuffer, std::vector<VkVariable*>);
		void addComputeToGraphicsBarriers(VkCommandBuffer commandBuffer, std::vector<VkVariable*>);

		// Resources for the compute part of the example
		struct {
			struct Semaphores {
				VkSemaphore ready{ 0L };
				VkSemaphore complete{ 0L };
			} semaphores;
		} compute;

		void bindPipeline();

		void setVkCommandBuffer(VkCommandBuffer cmdBuffer);
		void suspendInherentCmdBuffer(VkCommandBuffer cmdBuffer);
		void restoreInherentCmdBuffer();

	protected:
		void pushFormalParameter(const VkArgInfo& arg);
		void pushFormalConstant(const VkArgInfo& arg);

	private:
		VkPipelineShaderStageCreateInfo createComputeStage(std::string fileName);
		VkPushConstantRange buildPushRange(VkShaderStageFlags, const std::vector<VkArgInfo>&);
		std::vector<std::byte> buildPushBuf(const std::vector<VkVariable*>&);
		VkCommandBuffer commandBuffer() const;

		VkContext* ctx = nullptr;

		std::vector<VkArgInfo> mFormalParamters;
		std::vector<VkArgInfo> mFormalConstants;

		// Descriptor set pool
		// VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
		VkDescriptorSetLayout descriptorSetLayout {VK_NULL_HANDLE};
		VkDescriptorCache::LayoutHandle mLayoutHandle;

		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;
		
		VkQueue queue;

		std::optional<VkCommandBuffer> mCommandBuffer;
		std::optional<VkCommandBuffer> mOldCommandBuffer;

		std::vector<VkShaderModule> shaderModules;
	};

	template<typename... Args>
	VkProgram::VkProgram(Args... args)
	{
		ctx = VkSystem::instance()->currentContext();

		// Semaphores for graphics / compute synchronization
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(ctx->deviceHandle(), &semaphoreCreateInfo, nullptr, &compute.semaphores.ready));
		VK_CHECK_RESULT(vkCreateSemaphore(ctx->deviceHandle(), &semaphoreCreateInfo, nullptr, &compute.semaphores.complete));

		uint32_t nBuffer = 0;
		uint32_t nUniform = 0;
		std::initializer_list<VkArgInfo> argInfos{args...};
		for (auto arg : argInfos)
		{
			switch (arg.var_type)
			{
			case VariableType::DeviceBuffer:
				this->pushFormalParameter(arg);
				nBuffer++;
				break;
			case VariableType::Uniform:
				this->pushFormalParameter(arg);
				nUniform++;
				break;
			case VariableType::Constant:
				this->pushFormalConstant(arg);
				break;
			default:
				break;
			}
		}

		//Create descriptor set layout
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
		for (size_t i = 0; i < mFormalParamters.size(); i++)
		{
			if (mFormalParamters[i].var_type == VariableType::DeviceBuffer || mFormalParamters[i].var_type == VariableType::Uniform)
			{
				setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(VkVariable::descriptorType(mFormalParamters[i].var_type), VK_SHADER_STAGE_COMPUTE_BIT, i));
			}
		}

		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		mLayoutHandle = ctx->descriptorCache().queryLayoutHandle(descriptorLayout);
		descriptorSetLayout = ctx->descriptorCache().layout(mLayoutHandle).value();

		// VK_CHECK_RESULT(vkCreateDescriptorSetLayout(ctx->deviceHandle(), &descriptorLayout, nullptr, &descriptorSetLayout));


		// std::vector<VkDescriptorPoolSize> poolSizes;
		
		// if (nBuffer > 0)
		// 	poolSizes.push_back(vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nBuffer));

		// if (nUniform > 0)
		// 	poolSizes.push_back(vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nUniform));

		// if(poolSizes.size()) {
		// 	VkDescriptorPoolCreateInfo descriptorPoolInfo =
		// 		vks::initializers::descriptorPoolCreateInfo(poolSizes, std::max(poolSizes.size(), (std::size_t)1) * MaxDiscriptorSetNum);
		// 	VK_CHECK_RESULT(vkCreateDescriptorPool(ctx->deviceHandle(), &descriptorPoolInfo, nullptr, &descriptorPool));
		// }

		// Create a compute capable device queue
		vkGetDeviceQueue(ctx->deviceHandle(), ctx->computeQueueFamilyIndex(), 0, &queue);

		//VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
		//VK_CHECK_RESULT(vkCreateFence(ctx->deviceHandle(), &fenceInfo, nullptr, &mFence));
	}

	template<typename... Args>
	void VkProgram::enqueue(dim3 groupSize, Args... args)
	{
		std::initializer_list<VkVariable*> variables{const_cast<VkVariable*>((const VkVariable*)args)...};

#ifndef NDEBUG
		assert(mFormalParamters.size() == variables.size());
		constexpr std::array types { &typeid(std::decay_t<std::remove_pointer_t<decltype(args)>>)... };
		for (std::size_t i = 0; i < variables.size(); i++)
		{
			auto variable = *(variables.begin() + i);
			auto& p = mFormalParamters[i];
			assert(p.var_type == variable->type());
			if(p.var_type == VariableType::Constant && p.var_type == VariableType::Uniform) {
				assert(*p.type == *types[i]);
			}
		}
#endif // !NDEBUG

		auto cmd = commandBuffer();

		std::vector<VkVariable*> constArgs;
		std::vector<VkVariable*> bufferArgs;

		for (auto variable : variables)
		{
			switch (variable->type())
			{
			case VariableType::DeviceBuffer:
				bufferArgs.push_back(variable);
				// this->pushDeviceBuffer(variable);
				break;
			case VariableType::Uniform:
				//this->pushUniform(variable);
				break;
			case VariableType::Constant:
				constArgs.push_back(variable);
				break;
			default:
				break;
			}
		}

		{
			auto& compCtx = VkCompContext::current();
			auto descriptorSet = compCtx.allocDescriptorSet(mLayoutHandle);
			//Create descriptor set layout
			std::vector<VkWriteDescriptorSet> writeDescriptorSets;
			for (size_t i = 0; i < variables.size(); i++)
			{
				auto variable = *(variables.begin() + i);
				if ((variable->type() == VariableType::DeviceBuffer && variable->bufferSize() > 0) || variable->type() == VariableType::Uniform) {
					writeDescriptorSets.push_back(
						vks::initializers::writeDescriptorSet(descriptorSet, VkVariable::descriptorType(variable->type()), i, &variable->getDescriptor()));
				}
				if (variable->type() == VariableType::DeviceBuffer) {
					VkCompContext::current().registerBuffer(variable->bufferSp());
				}
			}

			vkUpdateDescriptorSets(ctx->deviceHandle(), static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, 0);
			writeDescriptorSets.clear();
		}
		
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

		{
			auto pushBuf = buildPushBuf(constArgs);
			vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBuf.size(), pushBuf.data());
		}

		vkCmdDispatch(cmd, groupSize.x, groupSize.y, groupSize.z);

		addComputeToComputeBarriers(cmd, bufferArgs);
	}

	template<typename... Args>
	void VkProgram::write(Args... args)
	{
		std::initializer_list<VkVariable*> variables{ args... };

#ifndef NDEBUG
		assert(mFormalParamters.size() == variables.size());
		constexpr std::array types { &typeid(std::decay_t<std::remove_pointer_t<decltype(args)>>)... };
		for (std::size_t i = 0; i < variables.size(); i++)
		{
			auto variable = *(variables.begin() + i);
			auto& p = mFormalParamters[i];
			assert(p.var_type == variable->type());
			if(p.var_type == VariableType::Constant && p.var_type == VariableType::Uniform) {
				assert(*p.type == *types[i]);
			}
		}
#endif // !NDEBUG

		for (auto variable : variables)
		{
			switch (variable->type())
			{
			case VariableType::DeviceBuffer:
		//		this->pushDeviceBuffer(variable);
				break;
			case VariableType::Uniform:
		//		this->pushUniform(variable);
				break;
			case VariableType::Constant:
		//		this->pushConstant(variable);
				break;
			default:
				break;
			}
		}

		{
			auto descriptorSet = ctx->descriptorCache().querySet(mLayoutHandle).value();
			//Create descriptor set layout
			std::vector<VkWriteDescriptorSet> writeDescriptorSets;
			for (size_t i = 0; i < variables.size(); i++)
			{
				auto variable = *(variables.begin() + i);
				if ((variable->type() == VariableType::DeviceBuffer && variable->bufferSize() > 0) || variable->type() == VariableType::Uniform) {
					writeDescriptorSets.push_back(
						vks::initializers::writeDescriptorSet(descriptorSet, VkVariable::descriptorType(variable->type()), i, &variable->getDescriptor()));
				}
			}

			vkUpdateDescriptorSets(ctx->deviceHandle(), static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
			writeDescriptorSets.clear();
		}
	}
}