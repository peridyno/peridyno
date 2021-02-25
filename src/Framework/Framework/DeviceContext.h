#pragma once
#include "Platform.h"
#include <list>
#include <memory>
#include "Framework/Module.h"

namespace dyno
{
/**
*  \brief Base class for simulation context
*
*  This class contains all shared data for a simulation algorithm.
*
*/
class DeviceContext : public Module
{
public:
	DeviceContext();
	virtual ~DeviceContext();

	void enable();

	bool setDevice(int i);
	int getDevice();

/*	template<typename T>
	std::shared_ptr< DeviceVariable<T> > allocDeviceVariable(std::string name, std::string description)
	{
		return allocVariable<T, DeviceType::GPU>(name, description);
	}

	template<typename T>
	std::shared_ptr< DeviceBuffer<T> > allocDeviceBuffer(std::string name, std::string description, int num)
	{
		return allocArrayBuffer<T, DeviceType::GPU>(name, description, num);
	}

	template<typename T>
	std::shared_ptr< DeviceVariable<T> > getDeviceVariable(std::string name)
	{
		return getVariable<T, DeviceType::GPU>(name);
	}

	template<typename T>
	std::shared_ptr< DeviceBuffer<T> > getDeviceBuffer(std::string name)
	{
		return getArrayBuffer<T, DeviceType::GPU>(name);
	}*/

public:
	int m_deviceID;
	int m_deviceNum;
	DeviceType m_deviceType;
		
	cudaStream_t stream;
};

}
