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

		Array3D(size_t nx, size_t ny, size_t nz)
		{
			this->resize(nx, ny, nz);
		};

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~Array3D() { };

		void resize(size_t nx, size_t ny, size_t nz);

		void reset();

		void clear();

		inline const T* begin() const { return m_data.data(); }

		inline size_t nx() const { return m_nx; }
		inline size_t ny() const { return m_ny; }
		inline size_t nz() const { return m_nz; }

		inline T operator () (const size_t i, const size_t j, const size_t k) const
		{
			return m_data[i + j * m_nx + k * m_nxy];
		}

		inline T& operator () (const size_t i, const size_t j, const size_t k)
		{
			return m_data[i + j * m_nx + k * m_nxy];
		}

		inline size_t index(const size_t i, const size_t j, const size_t k) const
		{
			return i + j * m_nx + k * m_nxy;
		}

		inline T operator [] (const size_t id) const
		{
			return m_data[id];
		}

		inline T& operator [] (const size_t id)
		{
			return m_data[id];
		}

		inline size_t size() const { return m_data.size(); }
		inline bool isCPU() const { return true; }
		inline bool isGPU() const { return false; }

		void assign(const Array3D<T, DeviceType::GPU>& src);
		void assign(const Array3D<T, DeviceType::CPU>& src);

	private:
		size_t m_nx = 0;
		size_t m_ny = 0;
		size_t m_nz = 0;
		size_t m_nxy = 0;
		std::vector<T>	m_data;
	};


	template<typename T>
	class Array3D<T, DeviceType::GPU>
	{
	public:
		Array3D() 
		{};

		Array3D(int nx, int ny, int nz)
		{
			this->resize(nx, ny, nz);
		};

		/*!
			*	\brief	Should not release data here, call Release() explicitly.
			*/
		~Array3D() { };

		void resize(const size_t nx, const size_t ny, const size_t nz);

		void reset();

		void clear();

		inline T* begin() const { return m_data; }

		DYN_FUNC inline size_t nx() const { return m_nx; }
		DYN_FUNC inline size_t ny() const { return m_ny; }
		DYN_FUNC inline size_t nz() const { return m_nz; }
		DYN_FUNC inline size_t pitch() const { return m_pitch_x; }
		
		DYN_FUNC inline T operator () (const int i, const int j, const int k) const
		{
			return m_data[i + j* m_pitch_x + k*m_nxy];
		}

		DYN_FUNC inline T& operator () (const int i, const int j, const int k)
		{
			return m_data[i + j* m_pitch_x + k*m_nxy];
		}

		DYN_FUNC inline size_t index(const int i, const int j, const int k) const
		{
			return i + j*m_pitch_x + k*m_nxy;
		}

		DYN_FUNC inline T operator [] (const int id) const
		{
			return m_data[id];
		}

		DYN_FUNC inline T& operator [] (const int id)
		{
			return m_data[id];
		}

		DYN_FUNC inline size_t size() const { return m_nxy * m_nz; }
		DYN_FUNC inline bool isCPU() const { return false; }
		DYN_FUNC inline bool isGPU() const { return true; }

		void assign(const Array3D<T, DeviceType::GPU>& src);
		void assign(const Array3D<T, DeviceType::CPU>& src);

	private:
		size_t m_nx = 0;
		size_t m_pitch_x = 0;

		size_t m_ny = 0;
		size_t m_nz = 0;
		size_t m_nxy = 0;
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