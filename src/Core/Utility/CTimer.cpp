#include "CTimer.h"

namespace dyno
{
	CTimer::CTimer()
	{
#if (defined _WIN32)
		QueryPerformanceFrequency(&timer_frequency_);
#endif
		start();
	}

	CTimer::~CTimer()
	{
	}

	void CTimer::start()
	{
#if (defined __unix__) || (defined __APPLE__)
		timeval tv;
		gettimeofday(&tv, 0);
		start_sec_ = tv.tv_sec;
		start_micro_sec_ = tv.tv_usec;
#elif (defined _WIN32)
		QueryPerformanceCounter(&start_count_);
#endif
	}

	void CTimer::stop()
	{
#if (defined __unix__) || (defined __APPLE__)
		timeval tv;
		gettimeofday(&tv, 0);
		stop_sec_ = tv.tv_sec;
		stop_micro_sec_ = tv.tv_usec;
#elif (defined _WIN32)
		QueryPerformanceCounter(&stop_count_);
#endif
	}

	double CTimer::getElapsedTime()
	{
#if (defined __unix__) || (defined __APPLE__)
		double elapsed_time = 1.0 * (stop_sec_ - start_sec_) + 1.0e-6 * (stop_micro_sec_ - start_micro_sec_);
		return elapsed_time;
#elif (defined _WIN32)
		double elapsed_time = static_cast<double>(stop_count_.QuadPart - start_count_.QuadPart) / static_cast<double>(timer_frequency_.QuadPart);
		return elapsed_time;
#endif
	}

} // end of namespace dyno
