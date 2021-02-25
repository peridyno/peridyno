#pragma once
#if (defined __unix__) || (defined __APPLE__)
#include <sys/time.h>
#elif (defined _WIN32)
#include <windows.h>
#endif

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

} //end of namespace dyno
