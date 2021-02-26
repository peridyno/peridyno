#pragma once
#include "Array/Array.h"
#include "Array/Array2D.h"
#include "Array/Array3D.h"
/*
*  This file implements all one-point functions on device array types (GArray, DeviceArray2D, DeviceArray3D, etc.)
*/
namespace dyno
{
	namespace Function1Pt
	{ 
		template<typename T, DeviceType dType1, DeviceType dType2>
		void copy(Array<T, dType1>& arr1, Array<T, dType2>& arr2)
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
			assert(g1.Size() == g2.Size() && g1.Nx()() == g2.Nx() && g2.Ny() == g2.Ny());
			size_t totalNum = g1.Size();
			if (g1.IsGPU() && g2.IsGPU())	(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyDeviceToDevice));
			else if (g1.IsCPU() && g2.IsGPU())	(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
			else if (g1.IsGPU() && g2.IsCPU())	(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyHostToDevice));
			else if (g1.IsCPU() && g2.IsCPU())	memcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T));
		}

		template<typename T, DeviceType dType1, DeviceType dType2>
		void copy(Array3D<T, dType1>& g1, Array3D<T, dType1>& g2)
		{
			assert(g1.Size() == g2.Size() && g1.Nx()() == g2.Nx() && g2.Ny() == g2.Ny() && g1.Nz() == g2.Nz());
			size_t totalNum = g1.Size();
			if (g1.IsGPU() && g2.IsGPU())	(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyDeviceToDevice));
			else if (g1.IsCPU() && g2.IsGPU())	(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyDeviceToHost));
			else if (g1.IsGPU() && g2.IsCPU())	(cudaMemcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T), cudaMemcpyHostToDevice));
			else if (g1.IsCPU() && g2.IsCPU())	memcpy(g1.GetDataPtr(), g2.GetDataPtr(), totalNum * sizeof(T));
		}

		template<typename T1, typename T2>
		void Length(GArray<T1>& lhs, GArray<T2>& rhs);


	}
}
