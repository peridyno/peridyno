#include "Utility.h"

TimeElapse::TimeElapse()
{
	this->tp = std::chrono::system_clock::now();
}

double TimeElapse::elapse()
{
	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - this->tp);
	return (double)(duration.count()) / 1000.0;
}

