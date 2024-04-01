#include "VkTransfer.h"

namespace dyno {
	inline int ScanSizeOfNextLevel(int size, int localSize)
	{
		return (size + localSize) / localSize;
	}

	template<typename T>
	void VkScan<T>::scan(std::vector<T>& inputData, uint ScanType) {
		VkDeviceArray<T> input;
		input.resize(inputData.size());
		vkTransfer(input, inputData);

		VkDeviceArray<T> output;
		output.resize(inputData.size());

		scan(output, input, ScanType);

		vkTransfer(inputData, output);

		output.clear();
	}

	template<typename T>
	void VkScan<T>::scan(VkDeviceArray<T>& output, const VkDeviceArray<T>& input, uint _ScanType)
	{
		VkCompContext::Holder holder;

		if(input.size() == 0) {
			return;
		}
		else if (input.size() == 1) {
			if(_ScanType == Type::Exclusive){
				std::vector<T> tmp{ 0 };
				vkTransfer(output, tmp);
				return;
			}
			else if (_ScanType == Type::Inclusive) {
				vkTransfer(output, input);
				return;
			}
		}


		uint localSize = 256;
		auto globalSize = input.size();

		std::vector<VkDeviceArray<T>> buffers;

		int n = globalSize;
		while (n > 1)
		{
			buffers.push_back(VkDeviceArray<T>(n));
			n = ScanSizeOfNextLevel(n, localSize);
		}

		buffers.push_back(VkDeviceArray<T>(1));

		vkTransfer(buffers[0], input);

		VkConstant<ScanParameters> push;
		push->n = buffers[0].size();
		push->ScanType = _ScanType;

		dim3 groupSize = vkDispatchSize(push->n, localSize);
	
 		mScan->begin();
 		mScan->enqueue(groupSize, &buffers[0], &buffers[0], &buffers[1], &push);
 		for (std::size_t i = 1; i < buffers.size() - 1; i++)
 		{
 			push->n = buffers[i].size();
 			push->ScanType = Type::Inclusive;
 			mScan->enqueue(vkDispatchSize(push->n, localSize), &buffers[i], &buffers[i], &buffers[i + 1], &push);
 		}
 		mScan->end();
 		auto fence = mScan->update(true);
 		mScan->wait(fence.value());

		VkConstant<int> num;
		mAdd->begin();
		for (std::size_t i = buffers.size() - 1; i > 0; i--)
		{
			num.setValue(buffers[i - 1].size());
			mAdd->enqueue(vkDispatchSize(num.getValue(), localSize), &buffers[i], &buffers[i - 1], &num);
		}
		mAdd->end();
		fence = mAdd->update(true);
		mAdd->wait(fence.value());

		vkTransfer(output, buffers[0]);

	}

	template<typename T>
	VkScan<T>::VkScan() {
		mScan = std::make_shared<VkProgram>(
			BUFFER(T),
			BUFFER(T),
			BUFFER(T),
			CONSTANT(ScanParameters));
		

		mAdd = std::make_shared<VkProgram>(
			BUFFER(T),
			BUFFER(T),
			CONSTANT(int));

		mAdd->load(getDynamicSpvFile<T>("shaders/glsl/core/Add.comp.spv"));
		mScan->load(getDynamicSpvFile<T>("shaders/glsl/core/Scan.comp.spv"));
	}

	template<typename T>
	VkScan<T>::~VkScan() {
		mScan = nullptr;
		mAdd = nullptr;
	}
}