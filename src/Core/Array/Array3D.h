#pragma once
#include "Platform.h"
#include <vector>

namespace dyno {
	template<typename T, DeviceType deviceType> class Array3D;

	template<typename T>
	class Array3D<T, DeviceType::CPU>
	{
	public:
		Array3D() {};

		Array3D(uint nx, uint ny, uint nz)
		{
			this->resize(nx, ny, nz);
		};

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~Array3D() { };

		void resize(uint nx, uint ny, uint nz);

		void reset();

		void clear();

		inline const std::vector<T>* handle() const { return &m_data; }
		inline std::vector<T>* handle() { return &m_data; }

		inline const T* begin() const { return m_data.data(); }

		inline uint nx() const { return m_nx; }
		inline uint ny() const { return m_ny; }
		inline uint nz() const { return m_nz; }

		inline T operator () (const uint i, const uint j, const uint k) const
		{
			return m_data[i + j * m_nx + k * m_nxy];
		}

		inline T& operator () (const uint i, const uint j, const uint k)
		{
			return m_data[i + j * m_nx + k * m_nxy];
		}

		inline size_t index(const uint i, const uint j, const uint k) const
		{
			return i + j * m_nx + k * m_nxy;
		}

		inline T operator [] (const uint id) const
		{
			return m_data[id];
		}

		inline T& operator [] (const uint id)
		{
			return m_data[id];
		}

		inline size_t size() const { return m_data.size(); }
		inline bool isCPU() const { return true; }
		inline bool isGPU() const { return false; }

		void assign(const T& val);
		void assign(uint nx, uint ny, uint nz, const T& val);

#ifndef NO_BACKEND
		void assign(const Array3D<T, DeviceType::GPU>& src);
#endif

		void assign(const Array3D<T, DeviceType::CPU>& src);

	private:
		uint m_nx = 0;
		uint m_ny = 0;
		uint m_nz = 0;
		uint m_nxy = 0;
		std::vector<T>	m_data;
	};

	template<typename T>
	void Array3D<T, DeviceType::CPU>::resize(uint nx, uint ny, uint nz)
	{
		m_data.clear();
		m_nx = nx;	m_ny = ny;	m_nz = nz; m_nxy = nx * ny;

		m_data.resize((size_t)nx * ny * nz);
	}

	template<typename T>
	void Array3D<T, DeviceType::CPU>::reset()
	{
		std::fill(m_data.begin(), m_data.end(), 0);
	}

	template<typename T>
	void Array3D<T, DeviceType::CPU>::clear()
	{
		m_nx = 0;
		m_ny = 0;
		m_nz = 0;
		m_data.clear();
	}

	template<typename T>
	void Array3D<T, DeviceType::CPU>::assign(uint nx, uint ny, uint nz, const T& val)
	{
		if (m_nx != nx || m_ny != ny || m_nz != nz) {
			this->resize(nx, ny, nz);
		}

		m_data.assign(m_data.size(), val);
	}

	template<typename T>
	void Array3D<T, DeviceType::CPU>::assign(const T& val)
	{
		m_data.assign(m_data.size(), val);
	}

	template<typename T>
	void Array3D<T, DeviceType::CPU>::assign(const Array3D<T, DeviceType::CPU>& src)
	{
		if (m_nx != src.size() || m_ny != src.size() || m_nz != src.size()) {
			this->resize(src.nx(), src.ny(), src.nz());
		}

		m_data.assign(src.m_data);
	}

	template<typename T>
	using CArray3D = Array3D<T, DeviceType::CPU>;
}

#ifdef CUDA_BACKEND
#include "Backend/Cuda/Array/Array3D.inl"
#endif

#ifdef VK_BACKEND
#include "Backend/Vulkan/Array/Array3D.inl"
#endif