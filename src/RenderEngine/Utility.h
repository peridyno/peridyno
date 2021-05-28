#pragma once

#include <chrono>

// a simple class to measure time elapse
class TimeElapse
{
public:
	TimeElapse();
	double elapse();

private:
	std::chrono::system_clock::time_point tp;
};

// get current framebuffer id
unsigned int GetCurrentFramebuffer();