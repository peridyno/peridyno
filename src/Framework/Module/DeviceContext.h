#pragma once
#include "Platform.h"
#include <list>
#include <memory>
#include "Module.h"

namespace dyno
{
/**
*  \brief Base class for simulation context
*
*  TODO: to support simulation on multi GPUs.
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

public:
	int m_deviceID;
	int m_deviceNum;
	DeviceType m_deviceType;
		
	cudaStream_t stream;
};

}
