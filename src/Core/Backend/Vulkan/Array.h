#pragma once
#include <memory>

#include "Platform.h"
#include "VkDeviceArray.h"

namespace px {

	template<typename T, DeviceType deviceType = DeviceType::GPU>
	class Array
	{
	public:
		Array();

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~Array() {};

		void resize(int size);

		/*!
		*	\brief	Clear all data to zero.
		*/
		void reset();

		/*!
		*	\brief	Free allocated memory.	Should be called before the object is deleted.
		*/
		void release();

	protected:
		void allocMemory();
		
	private:
		T* m_data;
		int m_totalNum;
	};

	template<typename T>
	using HostArray = Array<T, DeviceType::CPU>;

	template<typename T>
	using DeviceArray = Array<T, DeviceType::GPU>;
}

#include "Array.inl"
