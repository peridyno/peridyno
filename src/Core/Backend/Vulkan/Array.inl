#include "Array.h"

namespace px {

	template<typename T, DeviceType deviceType /*= DeviceType::GPU*/>
	Array<T, deviceType>::Array()
	{
		//assert(device);
		//mDevice = device;
	}

	template<typename T, DeviceType deviceType /*= DeviceType::GPU*/>
	void Array<T, deviceType>::resize(int size)
	{

	}
}
