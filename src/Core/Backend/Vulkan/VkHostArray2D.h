#pragma once
#include "VkVariable.h"
#include "VulkanBuffer.h"

namespace px {

	struct HostArray2DInfo
	{
		uint32_t nx;
		uint32_t ny;
		uint32_t nxy;
	};

	template<typename T>
	class VkHostArray2D : public VkVariable
	{

	public:
		VkHostArray2D() {};

		VkHostArray2D(uint32_t nx, uint32_t ny);

		~VkHostArray2D();

		uint32_t index(uint32_t i, uint32_t j);


		void resize(uint32_t nx, uint32_t ny, const T* data = nullptr);

		//inline uint32_t size() const { return m_num; }

		VariableType type() override;

		uint32_t bufferSize() override { return sizeof(T)*m_num; }

		void clear();

		HostArray2DInfo getInfo();

		inline uint32_t size() const { return m_num; }
		inline uint32_t nx() const { return m_nx; }
		inline uint32_t ny() const { return m_ny; }


		void* mapped();
		void unmap();

	private:
		uint32_t m_nx = 0;
		uint32_t m_ny = 0;

		uint32_t m_num = 0;
	};
}

#include "VkHostArray2D.inl"