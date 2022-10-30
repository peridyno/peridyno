#pragma once
#include "VkVariable.h"
#include "VulkanBuffer.h"

namespace px {

	struct Array3DInfo
	{
		uint32_t nx;
		uint32_t ny;
		uint32_t nz;
		uint32_t nxy;
	};

	template<typename T>
	class VkDeviceArray3D : public VkVariable
	{
	public:
		VkDeviceArray3D() {};

		VkDeviceArray3D(uint32_t nx, uint32_t ny, uint32_t nz);

		~VkDeviceArray3D();

		uint32_t index(uint32_t i, uint32_t j, uint32_t k);

		void resize(uint32_t nx, uint32_t ny, uint32_t nz, VkBufferUsageFlags usageFlags = 0);

		VariableType type() override;

		uint32_t bufferSize() override { return sizeof(T)*m_num; }

		void clear();

		Array3DInfo getInfo();

		inline uint32_t size() const { return m_num; }
		inline uint32_t nx() const { return m_nx; }
		inline uint32_t ny() const { return m_ny; }
		inline uint32_t nz() const { return m_nz; }

	private:
		uint32_t m_nx = 0;
		uint32_t m_ny = 0;
		uint32_t m_nz = 0;

		uint32_t m_num = 0;
	};
}

#include "VkDeviceArray3D.inl"