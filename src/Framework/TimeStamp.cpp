#include "TimeStamp.h"

namespace dyno {

	TimeStamp::~TimeStamp()
	{

	}

	void TimeStamp::mark()
	{
		static std::atomic<uint64_t> GlobalTickTime(0U);
		mTickTime = (uint64)++GlobalTickTime;
	}

}