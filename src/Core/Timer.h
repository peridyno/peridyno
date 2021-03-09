#pragma once
#if (defined __unix__) || (defined __APPLE__)
#include <sys/time.h>
#elif (defined _WIN32)
#include <windows.h>
#endif

#include <cuda_runtime.h>
#include <iostream>

namespace dyno 
{
	class CTimer
	{
	public:
		CTimer();
		~CTimer();
		void start();
		void stop();
		double getElapsedTime();
	protected:
#if (defined __unix__) || (defined __APPLE__)
		long start_sec_, stop_sec_, start_micro_sec_, stop_micro_sec_;
#elif (defined _WIN32)
		LARGE_INTEGER timer_frequency_;
		LARGE_INTEGER start_count_, stop_count_;
#endif
	};

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
} //end of namespace dyno
