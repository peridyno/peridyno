#include "DeviceContext.h"
#include <cuda_runtime.h>

namespace dyno {

DeviceContext::DeviceContext()
{
	m_deviceNum = -1;
	m_deviceID = -1;
	m_deviceType = DeviceType::GPU;

	cudaGetDeviceCount(&m_deviceNum);
	if (m_deviceNum > 0)
	{
		setDevice(0);
	}
	else
	{
		std::cout << "No device available!" << std::endl;
	}
}

DeviceContext::~DeviceContext()
{

}

void DeviceContext::enable()
{
	cudaSetDevice(m_deviceID);
}

bool DeviceContext::setDevice(int i)
{
	if (i >= m_deviceNum) return false;

	m_deviceID = i;
	cudaSetDevice(m_deviceID);
	return true;
}

int DeviceContext::getDevice()
{
	return m_deviceID;
}

}