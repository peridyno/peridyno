#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <cstring>
#include "Platform.h"

namespace dyno {

#define INVALID -1

	template<typename T, DeviceType deviceType = DeviceType::GPU>
	class Array3D
	{
	public:
		Array3D() 
			: m_nx(0)
			, m_ny(0)
			, m_nz(0)
			, m_nxy(0)
			, m_totalNum(0)
			, m_data(NULL)
		{};

		Array3D(int nx, int ny, int nz)
			: m_nx(nx)
			, m_ny(ny)
			, m_nz(nz)
			, m_nxy(nx*ny)
			, m_totalNum(nx*ny*nz)
			, m_data(NULL)
		{
			AllocMemory();
		};

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~Array3D() { };

		void resize(int nx, int ny, int nz);

		void reset();

		void clear();

		inline T*		data() { return m_data; }

		DYN_FUNC inline int nx() { return m_nx; }
		DYN_FUNC inline int ny() { return m_ny; }
		DYN_FUNC inline int nz() { return m_nz; }
		
		DYN_FUNC inline T operator () (const int i, const int j, const int k) const
		{
			return m_data[i + j*m_nx + k*m_nxy];
		}

		DYN_FUNC inline T& operator () (const int i, const int j, const int k)
		{
			return m_data[i + j*m_nx + k*m_nxy];
		}

		DYN_FUNC inline int index(const int i, const int j, const int k)
		{
			return i + j*m_nx + k*m_nxy;
		}

		DYN_FUNC inline T operator [] (const int id) const
		{
			return m_data[id];
		}

		DYN_FUNC inline T& operator [] (const int id)
		{
			return m_data[id];
		}

		DYN_FUNC inline int size() const { return m_totalNum; }
		DYN_FUNC inline bool isCPU() const { return deviceType == DeviceType::CPU; }
		DYN_FUNC inline bool isGPU() const { return deviceType == DeviceType::GPU; }

	public:
		void AllocMemory();

	private:
		int m_nx;
		int m_ny;
		int m_nz;
		int m_nxy;
		int m_totalNum;
		T*	m_data;
	};

	template<typename T, DeviceType deviceType>
	void Array3D<T, deviceType>::resize(int nx, int ny, int nz)
	{
		if (NULL != m_data) clear();
		m_nx = nx;	m_ny = ny;	m_nz = nz;	m_nxy = m_nx*m_ny;	m_totalNum = m_nxy*m_nz;
		AllocMemory();
	}

	template<typename T, DeviceType deviceType>
	void Array3D<T, deviceType>::reset()
	{
		switch (deviceType)
		{
		case CPU:
			memset((void*)m_data, 0, m_totalNum * sizeof(T));
			break;
		case GPU:
			cudaMemset(m_data, 0, m_totalNum * sizeof(T));
			break;
		default:
			break;
		}
	}

	template<typename T, DeviceType deviceType>
	void Array3D<T, deviceType>::clear()
	{
		if (m_data != NULL)
		{
			switch (deviceType)
			{
			case CPU:
				delete[]m_data;
				break;
			case GPU:
				(cudaFree(m_data));
				break;
			default:
				break;
			}
		}

		m_data = NULL;
		m_nx = 0;
		m_ny = 0;
		m_nz = 0;
		m_nxy = 0;
		m_totalNum = 0;
	}

	template<typename T, DeviceType deviceType>
	void Array3D<T, deviceType>::AllocMemory()
	{
		switch (deviceType)
		{
		case CPU:
			m_data = new T[m_totalNum];
			break;
		case GPU:
			(cudaMalloc((void**)&m_data, m_totalNum * sizeof(T)));
			break;
		default:
			break;
		}

		reset();
	}

	template<typename T>
	using CArray3D = Array3D<T, DeviceType::CPU>;

	template<typename T>
	using GArray3D = Array3D<T, DeviceType::GPU>;

	typedef GArray3D<float>	Grid1f;
	typedef GArray3D<float3> Grid3f;
	typedef GArray3D<bool> Grid1b;
}
