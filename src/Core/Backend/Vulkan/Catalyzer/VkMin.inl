
#include "VkTransfer.h"
#include "VkConstant.h"

namespace dyno {
	template<typename T>
	T VkMin<T>::reduce(const std::vector<T>& input)
	{
		VkHostArray<T> hostArray;
		hostArray.resize(input.size(), input.data());

		VkDeviceArray<T> deviceArray;
		deviceArray.resize(input.size());

		vkTransfer(deviceArray, hostArray);

		T ret = reduce(deviceArray);

		deviceArray.clear();
		hostArray.clear();

		return ret;
	}

	template<typename T>
	T VkMin<T>::reduce(const VkDeviceArray<T>& input)
	{
		VkCompContext::Holder holder;

		unsigned int localSize = 256;
		auto globalSize = input.size();

		std::vector<VkDeviceArray<T>> buffers;
		buffers.push_back(input);

		auto nextLevelSize = [](int input, int local) -> int {
			return (input + local) / local;
		};

		int n = nextLevelSize(globalSize, localSize);
		while (n > 1)
		{
			buffers.push_back(VkDeviceArray<T>(n));
			n = nextLevelSize(n, localSize);
		}

		buffers.push_back(VkDeviceArray<T>(1));

		VkConstant<int> num;
		VkConstant<T> init_val {std::numeric_limits<T>::max()};

		mReduceKernel->begin();

		for (std::size_t i = 0; i < buffers.size() - 1; i++)
		{
			num.setValue(buffers[i].size());
			dim3 groupSize = vkDispatchSize(num.getValue(), localSize);
			mReduceKernel->enqueue(groupSize, &buffers[i], &buffers[i + 1], &num, &init_val);
		}

		mReduceKernel->end();

		mReduceKernel->update(true);
		mReduceKernel->wait();

		std::vector<T> ret(1);
		vkTransfer(ret, buffers.back());

		T sum = ret[0];

		ret.clear();
		return sum;
	}

	template<typename T>
	VkMin<T>::~VkMin() {
		mReduceKernel = nullptr;
	}

	template<typename T>
	VkMin<T>::VkMin() {
		mReduceKernel = std::make_shared<VkProgram>(
			BUFFER(T),
			BUFFER(T),
			CONSTANT(int),
			CONSTANT(T));

		mReduceKernel->load(getDynamicSpvFile<T>("shaders/glsl/core/Min.comp.spv"));
	}
}


namespace dyno {
	template<typename T>
	uint VkMinElement<T>::reduce(const std::vector<T>& input)
	{
		VkHostArray<T> hostArray;
		hostArray.resize(input.size(), input.data());

		VkDeviceArray<T> deviceArray;
		deviceArray.resize(input.size());

		vkTransfer(deviceArray, hostArray);

		T ret = reduce(deviceArray);

		deviceArray.clear();
		hostArray.clear();

		return ret;
	}

	template<typename T>
	uint VkMinElement<T>::reduce(const VkDeviceArray<T>& input)
	{
		VkCompContext::Holder holder;

		unsigned int localSize = 256;
		auto globalSize = input.size();

		std::vector<VkDeviceArray<uint>> buffers;
		buffers.push_back(VkDeviceArray<uint>(input.size()));

		auto nextLevelSize = [](int input, int local) -> int {
			return (input + local) / local;
		};

		int n = nextLevelSize(globalSize, localSize);
		while (n > 1)
		{
			buffers.push_back(VkDeviceArray<uint>(n));
			n = nextLevelSize(n, localSize);
		}

		buffers.push_back(VkDeviceArray<uint>(1));

		VkConstant<uint> num;
		VkConstant<uint> value_num {input.size()};
		VkConstant<int> first_run {1};
		VkConstant<T> init_val {std::numeric_limits<T>::max()};

		mReduceKernel->begin();

		for (std::size_t i = 0; i < buffers.size() - 1; i++)
		{
			first_run.setValue(i == 0);
			num.setValue(buffers[i].size());
			dim3 groupSize = vkDispatchSize(num.getValue(), localSize);
			mReduceKernel->enqueue(groupSize, &buffers[i], &buffers[i + 1], &input, &num, &value_num, &first_run, &init_val);
		}

		mReduceKernel->end();

		mReduceKernel->update(true);
		mReduceKernel->wait();

		std::vector<uint> ret(1);
		vkTransfer(ret, buffers.back());

		T sum = ret[0];

		ret.clear();
		return sum;
	}

	template<typename T>
	VkMinElement<T>::~VkMinElement() {
		mReduceKernel = nullptr;
	}

	template<typename T>
	VkMinElement<T>::VkMinElement() {
		mReduceKernel = std::make_shared<VkProgram>(
			BUFFER(uint),
			BUFFER(uint),
			BUFFER(T),
			CONSTANT(uint),
			CONSTANT(uint),
			CONSTANT(int),
			CONSTANT(T));

		mReduceKernel->load(getDynamicSpvFile<T>("shaders/glsl/core/MinElement.comp.spv"));
	}
}