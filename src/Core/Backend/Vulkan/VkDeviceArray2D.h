#pragma once
#include "VkVariable.h"
#include "VulkanBuffer.h"

namespace dyno {

	struct Array2DInfo
	{
		uint32_t nx;
		uint32_t ny;
	};

	template<typename T>
	class VkDeviceArray2D : public VkVariable
	{

	public:
		VkDeviceArray2D() {};

		VkDeviceArray2D(uint32_t nx, uint32_t ny);

		~VkDeviceArray2D();

		void resize(uint32_t nx, uint32_t ny, VkBufferUsageFlags usageFlags = 0);


		VariableType type() override;

		uint32_t bufferSize() override { return sizeof(T)*m_num; }

		void clear();

		Array2DInfo getInfo();

		inline uint32_t size() const { return m_num; }
		inline uint32_t nx() const { return m_nx; }
		inline uint32_t ny() const { return m_ny; }

	private:
		uint32_t m_nx = 0;
		uint32_t m_ny = 0;

		uint32_t m_num = 0;
	};
}

#include "VkDeviceArray2D.inl"