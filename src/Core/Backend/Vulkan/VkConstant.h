#pragma once
#include "VkVariable.h"

namespace dyno {

	template<typename T>
	class VkConstant : public VkVariable
	{
	public:
		VkConstant();
		VkConstant(T val);
		~VkConstant();

		void setValue(const T val);
		T getValue() const;

		VariableType type() override;

		uint32_t bufferSize() override { return sizeof(T); }

		void* data() const override { return (void*)&mVal; }

		const T* operator->() const {
			return &mVal;
		}

		T* operator->() {
			return &mVal;
		}

	protected:
		T mVal;
	};
}

#include "VkConstant.inl"