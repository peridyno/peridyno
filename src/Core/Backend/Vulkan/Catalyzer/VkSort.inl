
#include "VkTransfer.h"

namespace dyno {
	
	inline int ComputeAddZeroSize(int OriginalSize) {
		int AddZeroSize = 2;
		while (AddZeroSize < OriginalSize) {
			AddZeroSize *= 2;
		}

		return AddZeroSize;
	}

	template<typename T>
	void VkSort<T>::sort(std::vector<T> &data, uint32_t SortType) {
		uint32_t dSize = data.size();

		VkDeviceArray<T> kArray;
		kArray.resize(data.size());
		vkTransfer(kArray, data);

		// sort dArray into a non-descending order
		sort(kArray, SortType);

		std::vector<T> ouputData(data.size());
		vkTransfer(ouputData, kArray);

		data.swap(ouputData);
	};


	

	template<typename T>
	void VkSort<T>::sort(VkDeviceArray<T>& data, uint32_t SortType)
	{
		int AddZeroSize = ComputeAddZeroSize(data.size());
		uint32_t globalSize = AddZeroSize;

		uint32_t workgroup_size_x;

		if (AddZeroSize < 128) {
			workgroup_size_x = AddZeroSize / 2;
		}
		else {
			workgroup_size_x = 64;
		}

		VkUniform<Parameters> uniformParam;

		VkDeviceArray<T> AddZeroArray;
		AddZeroArray.resize(AddZeroSize);

		const uint32_t workgroup_count = globalSize / (workgroup_size_x * 2);

		uint32_t h = workgroup_size_x * 2;

		dim3 groupSize;
		groupSize.x = workgroup_count;

		Parameters param;

		// SortType UP 0/ DOWN 1
		param.SortType = SortType;
        	param.srcSize = AddZeroArray.size();
        	param.dstSize = data.size();
		//AddZeroArray: add 0 elements to data array 
		param.h = data.size();
		param.algorithm = Parameters::addZero;
		uniformParam.setValue(param);
		mSortKernel->flush(groupSize, &data, &uniformParam, &AddZeroArray);

		//TODO: replace flush with enqueue
		param.h = h;
		param.algorithm = Parameters::eLocalBms;
		uniformParam.setValue(param);
		mSortKernel->flush(groupSize, &AddZeroArray, &uniformParam, &data);

		// we must now double h, as this happens before every flip
		h *= 2;

		for (; h <= globalSize; h *= 2) {

			param.h = h;
            		param.srcSize = data.size();
            		param.dstSize = AddZeroArray.size();
			//param.n = globalSize;
			param.algorithm = Parameters::eBigFlip;
			uniformParam.setValue(param);
			mSortKernel->flush(groupSize, &AddZeroArray, &uniformParam, &data);

			for (uint32_t hh = h / 2; hh > 1; hh /= 2) {
				if (hh <= workgroup_size_x * 2) {
					param.h = hh;
					//param.n = globalSize;
					param.algorithm = Parameters::eLocalDisperse;
					uniformParam.setValue(param);
					mSortKernel->flush(groupSize, &AddZeroArray, &uniformParam, &data);
					break;
				}
				else {
					param.h = hh;
					//param.n = globalSize;
					param.algorithm = Parameters::eBigDisperse;
					uniformParam.setValue(param);
					mSortKernel->flush(groupSize, &AddZeroArray, &uniformParam, &data);
				}
			}
		}

		//data subtract zero element
		param.h = h;
		param.algorithm = Parameters::subtractZero;
		uniformParam.setValue(param);
		mSortKernel->flush(groupSize, &data, &uniformParam, &AddZeroArray);
	}

	template<typename T>
	void VkSort<T>::sort_by_key(std::vector<T>& keys, std::vector<T>& values, uint32_t SortType) {
		assert(keys.size() == values.size());

		uint32_t dSize = keys.size();

		VkDeviceArray<T> kArray;
		kArray.resize(keys.size());
		vkTransfer(kArray, keys);

		VkDeviceArray<T> vArray;
		vArray.resize(values.size());
		vkTransfer(vArray, values);

		// sort dArray into a non-descending order
		sort_by_key(kArray, vArray, SortType);

		std::vector<T> ouputKData(keys.size());
		vkTransfer(ouputKData, kArray);

		std::vector<T> ouputVData(values.size());
		vkTransfer(ouputVData, vArray);

		keys.swap(ouputKData);
		values.swap(ouputVData);
	}
	
	

	template<typename T>
	void VkSort<T>::sort_by_key(VkDeviceArray<T>& keys, VkDeviceArray<T>& values, uint32_t SortType)
	{
		assert(keys.size() == values.size());
		int AddZeroSize = ComputeAddZeroSize(keys.size());

		uint32_t  workgroup_size_x;
		if (AddZeroSize < 128) {
			workgroup_size_x = AddZeroSize / 2;
		}
		else {
			workgroup_size_x = 64;
		}

		uint32_t globalSize = AddZeroSize;

		VkUniform<Parameters> uniformParam;

		VkDeviceArray<T> KeysAddZeroArray;
		KeysAddZeroArray.resize(AddZeroSize);
		VkDeviceArray<T> ValuesAddZeroArray;
		ValuesAddZeroArray.resize(AddZeroSize);

		const uint32_t workgroup_count = globalSize / (workgroup_size_x * 2);

		uint32_t h = workgroup_size_x * 2;

		dim3 groupSize;
		groupSize.x = workgroup_count;

		Parameters param;

		// SortType UP 0/ DOWN 1
		param.SortType = SortType;

		//AddZeroArray: add 0 elements to data array 
		param.h = keys.size();
        	param.srcSize = KeysAddZeroArray.size();
        	param.dstSize = keys.size();
		//param.h = h;
		param.algorithm = Parameters::addZero;
		uniformParam.setValue(param);
		mSortByKeyKernel->flush(groupSize, &keys, &values, &uniformParam, &KeysAddZeroArray, &ValuesAddZeroArray);

		//TODO: replace flush with enqueue
		param.h = h;
        	param.srcSize = keys.size();
        	param.dstSize = KeysAddZeroArray.size();
		param.algorithm = Parameters::eLocalBms;
		uniformParam.setValue(param);
		mSortByKeyKernel->flush(groupSize, &KeysAddZeroArray, &ValuesAddZeroArray, &uniformParam, &keys, &values);

		// we must now double h, as this happens before every flip
		h *= 2;

		//if the data is small, reduce the group

		for (; h <= globalSize; h *= 2) {
			param.h = h;
            		param.srcSize = keys.size();
            		param.dstSize = KeysAddZeroArray.size();
			//param.n = globalSize;
			param.algorithm = Parameters::eBigFlip;
			uniformParam.setValue(param);
			mSortByKeyKernel->flush(groupSize, &KeysAddZeroArray, &ValuesAddZeroArray, &uniformParam, &keys, &values);

			for (uint32_t hh = h / 2; hh > 1; hh /= 2) {
				if (hh <= workgroup_size_x * 2) {
					param.h = hh;
					//param.n = globalSize;
					param.algorithm = Parameters::eLocalDisperse;
					uniformParam.setValue(param);
					mSortByKeyKernel->flush(groupSize, &KeysAddZeroArray, &ValuesAddZeroArray, &uniformParam, &keys, &values);
					break;
				}
				else {
					param.h = hh;
					//param.n = globalSize;
					param.algorithm = Parameters::eBigDisperse;
					uniformParam.setValue(param);
					mSortByKeyKernel->flush(groupSize, &KeysAddZeroArray, &ValuesAddZeroArray, &uniformParam, &keys, &values);
				}
			}
		}

		//data subtract zero element
		param.h = h;
        	param.srcSize = KeysAddZeroArray.size();
        	param.dstSize = keys.size();
		param.algorithm = Parameters::subtractZero;
		uniformParam.setValue(param);
		mSortByKeyKernel->flush(groupSize, &keys, &values, &uniformParam, &KeysAddZeroArray, &ValuesAddZeroArray);

		KeysAddZeroArray.clear();
		ValuesAddZeroArray.clear();
	}

	template<typename T>
	VkSort<T>::VkSort() {

		mSortKernel = std::make_shared<VkProgram>(
			BUFFER(T),
			UNIFORM(Parameters),
			BUFFER(T));
		
		mSortByKeyKernel = std::make_shared<VkProgram>(
			BUFFER(T),
			BUFFER(T),
			UNIFORM(Parameters),
			BUFFER(T),
			BUFFER(T));

		mSortKernel->load(getDynamicSpvFile<T>("shaders/glsl/core/Sort.comp.spv"));
		mSortByKeyKernel->load(getDynamicSpvFile<T>("shaders/glsl/core/SortByKey.comp.spv"));
	}

	template<typename T>
	VkSort<T>::~VkSort() {
		mSortKernel = nullptr;
		mSortByKeyKernel = nullptr;
	}
}
