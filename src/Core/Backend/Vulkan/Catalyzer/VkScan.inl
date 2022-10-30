#include "VkTransfer.h"

namespace px {
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
	}

	template<typename T>
	void VkScan<T>::scan(VkDeviceArray<T>& output, const VkDeviceArray<T>& input, uint _ScanType)
	{
		assert(input.size() > 0);
		if (input.size() == 1) {
			if(_ScanType == EXCLUSIVESCAN){
				std::vector<T> tmp{ 0 };
				vkTransfer(output, tmp);
				return;
			}
			else if (_ScanType == INCLUSIVESCAN) {
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

		VkUniform<ScanParameters> uniformParam;
		ScanParameters sp;
		sp.n = buffers[0].size();
		sp.ScanType = _ScanType;

		dim3 groupSize = vkDispatchSize(sp.n, localSize);
	

		uniformParam.setValue(sp);
		mScan->flush(groupSize, &buffers[0], &buffers[0], &buffers[1], &uniformParam);

		mScan->begin();
		for (std::size_t i = 1; i < buffers.size() - 1; i++)
		{
			sp.n = buffers[i].size();
			dim3 groupSizeScan = vkDispatchSize(sp.n, localSize);
			sp.ScanType = INCLUSIVESCAN;

			uniformParam.setValue(sp);
			mScan->enqueue(groupSizeScan, &buffers[i], &buffers[i], &buffers[i + 1], &uniformParam);
		}
		mScan->end();
		mScan->update(true);
		mScan->wait();

		VkConstant<int> num;
		mAdd->begin();
		for (std::size_t i = 1; i < buffers.size() - 1; i++)
		{
			num.setValue(buffers[i - 1].size());
			dim3 groupSizeAdd = vkDispatchSize(num.getValue(), localSize);
			mAdd->enqueue(groupSizeAdd, &buffers[i], &buffers[i - 1], &num);
		}
		mAdd->end();
		mAdd->update(true);
		mAdd->wait();

		vkTransfer(output, buffers[0]);
		for (std::size_t i = 0; i < buffers.size(); i++)
		{
			buffers[i].clear();
		}
	}

	template<typename T>
	VkScan<T>::VkScan() {
		mScan = std::make_shared<VkProgram>(
			BUFFER(T),
			BUFFER(T),
			BUFFER(T),
			UNIFORM(ScanParameters));
		

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