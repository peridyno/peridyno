#include "GTimer.h"
#include <ctime>
#include <iostream>

namespace dyno {

	GTimer::GTimer()
	{
		milliseconds = 0.0f;
		cudaEventCreate(&m_start);
		cudaEventCreate(&m_stop);
	}

	GTimer::~GTimer()
	{
	}

	void GTimer::start()
	{
		cudaEventRecord(m_start, 0);
	}

	void GTimer::stop()
	{
		cudaEventRecord(m_stop, 0);
		cudaEventSynchronize(m_stop);
		cudaEventElapsedTime(&milliseconds, m_start, m_stop);
	}

	float GTimer::getEclipsedTime()
	{
		return milliseconds;
	}

	void GTimer::outputString(char* str)
	{
		std::cout << str << ": " << getEclipsedTime() << std::endl;
	}

}