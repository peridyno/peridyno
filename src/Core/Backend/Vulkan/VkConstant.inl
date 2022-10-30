#include "VkConstant.h"

namespace px {

	template<typename T>
	VkConstant<T>::VkConstant()
		: VkVariable()
	{
	}

	template<typename T>
	VkConstant<T>::VkConstant(T val)
	{
		mVal = val;
	}

	template<typename T>
	VkConstant<T>::~VkConstant()
	{

	}

	template<typename T>
	void VkConstant<T>::setValue(const T val)
	{
		mVal = val;
	}

	template<typename T>
	T VkConstant<T>::getValue()
	{
		return mVal;
	}

	template<typename T>
	VariableType VkConstant<T>::type()
	{
		return VariableType::Constant;
	}


}