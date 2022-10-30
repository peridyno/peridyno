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
		void assign(const Array3D<T, DeviceType::GPU>& src);
		void assign(const Array3D<T, DeviceType::CPU>& src);

	private:
		uint m_nx = 0;
		uint m_ny = 0;
		uint m_nz = 0;
		uint m_nxy = 0;
		std::vector<T>	m_data;
	};

	template<typename T>
	class Array3D<T, DeviceType::GPU>
	{
	public:
		Array3D() 
		{};

		Array3D(uint nx, uint ny, uint nz)
		{
			this->resize(nx, ny, nz);
		};

		/*!
			*	\brief	Should not release data here, call Release() explicitly.
			*/
		~Array3D() { };

		void resize(const uint nx, const uint ny, const uint nz);

		void reset();

		void clear();

		inline T* begin() const { return m_data; }

		DYN_FUNC inline uint nx() const { return m_nx; }
		DYN_FUNC inline uint ny() const { return m_ny; }
		DYN_FUNC inline uint nz() const { return m_nz; }
		DYN_FUNC inline uint pitch() const { return m_pitch_x; }
		
		DYN_FUNC inline T operator () (const int i, const int j, const int k) const
		{	
			char* addr = (char*)m_data;
			addr += (j * m_pitch_x + k * m_nxy);
			return ((T*)addr)[i];
		}

		DYN_FUNC inline T& operator () (const int i, const int j, const int k)
		{
			char* addr = (char*)m_data;
			addr += (j * m_pitch_x + k * m_nxy);
			return ((T*)addr)[i];
		}

		DYN_FUNC inline T operator [] (const int id) const
		{
			return m_data[id];
		}

		DYN_FUNC inline T& operator [] (const int id)
		{
			return m_data[id];
		}

		DYN_FUNC inline size_t size() const { return m_nx * m_ny * m_nz; }
		DYN_FUNC inline bool isCPU() const { return false; }
		DYN_FUNC inline bool isGPU() const { return true; }

		void assign(const Array3D<T, DeviceType::GPU>& src);
		void assign(const Array3D<T, DeviceType::CPU>& src);

	private:
		uint m_nx = 0;
		uint m_pitch_x = 0;

		uint m_ny = 0;
		uint m_nz = 0;
		uint m_nxy = 0;
		T*	m_data = nullptr;
	};

	template<typename T>
	using CArray3D = Array3D<T, DeviceType::CPU>;

	template<typename T>
	using DArray3D = Array3D<T, DeviceType::GPU>;

	typedef DArray3D<float>	Grid1f;
	typedef DArray3D<float3> Grid3f;
	typedef DArray3D<bool> Grid1b;
}

#include "Array3D.inl"