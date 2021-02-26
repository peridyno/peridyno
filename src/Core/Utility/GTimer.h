#pragma once
#include <cuda_runtime.h>

namespace dyno 
{
	class GTimer
	{
	private:
		cudaEvent_t m_start, m_stop;

		float milliseconds;

	public:
		GTimer();
		~GTimer();

		void start();
		void stop();

		float getEclipsedTime();

		void outputString(char* str);
	};
}




