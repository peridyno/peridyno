#pragma once
#include "Array/Array.h"
#include "Array/Array2D.h"
#include "Array/Array3D.h"
/*
*  This file implements all one-point functions on device array types (GArray, GArray2D, GArray3D, etc.)
*/
namespace dyno
{
	namespace Function1Pt
	{ 
		template<typename T, DeviceType dType1, DeviceType dType2>
		void copy(Array<T, dType1>& arr1, const Array<T, dType2>& arr2)
		{
			assert(arr1.size() == arr2.size());
			size_t totalNum = arr1.size();
			if (arr1.isGPU() && arr2.isGPU())	(cudaMemcpy(arr1.begin(), arr2.begin(), totalNum * sizeof(T), cudaMemcpyDeviceToDevice));
			else if (arr1.isCPU() && arr2.isGPU())	(cudaMemcpy(arr1.begin(), arr2.begin(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
			else if (arr1.isGPU() && arr2.isCPU())	(cudaMemcpy(arr1.begin(), arr2.begin(), totalNum * sizeof(T), cudaMemcpyHostToDevice));
			else if (arr1.isCPU() && arr2.isCPU())	memcpy(arr1.begin(), arr2.begin(), totalNum * sizeof(T));
		}

		template<typename T, DeviceType deviceType>
		void copy(Array<T, deviceType>& arr, std::vector<T>& vec)
		{
			assert(vec.size() == arr.size());
			size_t totalNum = arr.size();
			switch (deviceType)
			{
			case CPU:
				memcpy(arr.begin(), &vec[0], totalNum * sizeof(T));
				break;
			case GPU:
				(cudaMemcpy(arr.begin(), &vec[0], totalNum * sizeof(T), cudaMemcpyHostToDevice));
				break;
			default:
				break;
			}
		}

		template<typename T, DeviceType dType1, DeviceType dType2>
		void copy(Array2D<T, dType1>& g1, Array2D<T, dType1>& g2)
		{
			assert(g1.size() == g2.size() && g1.nx()() == g2.nx() && g2.ny() == g2.ny());
			size_t totalNum = g1.size();
			if (g1.isGPU() && g2.isGPU())	(cudaMemcpy(g1.getDataPtr(), g2.data(), totalNum * sizeof(T), cudaMemcpyDeviceToDevice));
			else if (g1.isCPU() && g2.isGPU())	(cudaMemcpy(g1.data(), g2.data(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
			else if (g1.isGPU() && g2.isCPU())	(cudaMemcpy(g1.data(), g2.data(), totalNum * sizeof(T), cudaMemcpyHostToDevice));
			else if (g1.isCPU() && g2.isCPU())	memcpy(g1.data(), g2.data(), totalNum * sizeof(T));
		}

		template<typename T, DeviceType dType1, DeviceType dType2>
		void copy(Array3D<T, dType1>& g1, Array3D<T, dType1>& g2)
		{
			assert(g1.size() == g2.size() && g1.nx()() == g2.nx() && g2.ny() == g2.ny() && g1.nz() == g2.nz());
			size_t totalNum = g1.size();
			if (g1.isGPU() && g2.isGPU())	(cudaMemcpy(g1.data(), g2.data(), totalNum * sizeof(T), cudaMemcpyDeviceToDevice));
			else if (g1.isCPU() && g2.isGPU())	(cudaMemcpy(g1.data(), g2.data(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
			else if (g1.isGPU() && g2.isCPU())	(cudaMemcpy(g1.data(), g2.data(), totalNum * sizeof(T), cudaMemcpyHostToDevice));
			else if (g1.isCPU() && g2.isCPU())	memcpy(g1.data(), g2.data(), totalNum * sizeof(T));
		}

		template<typename T1, typename T2>
		void Length(GArray<T1>& lhs, GArray<T2>& rhs);


	}
}
