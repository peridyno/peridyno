#pragma once
#if (defined __unix__) || (defined __APPLE__)
#include <sys/time.h>
#elif (defined _WIN32)
#include <windows.h>
#endif

#include "Platform.h"

#ifdef CUDA_BACKEND
	#include <cuda_runtime.h>
#endif
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

		/**
		 * @brief return the elapsed time in (ms)
		 */
		double getElapsedTime();
		void outputString(char* str);
	protected:
#if (defined __unix__) || (defined __APPLE__)
		long start_sec_, stop_sec_, start_micro_sec_, stop_micro_sec_;
#elif (defined _WIN32)
		LARGE_INTEGER timer_frequency_;
		LARGE_INTEGER start_count_, stop_count_;
#endif
	};

#ifdef CUDA_BACKEND
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

		float getElapsedTime();

		void outputString(char* str);
	};
#endif
} //end of namespace dyno
