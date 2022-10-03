#include "DebugInfo.h"
#include "Log.h"

#include <sstream>

namespace dyno 
{
	void DebugInfo::updateImpl()
	{
		this->print();
	}

	void PrintInt::print()
	{
		std::ostringstream oss;
		oss << this->inInt()->getData();

		Log::sendMessage(Log::Info, oss.str());
	}

	void PrintUnsigned::print()
	{
		std::ostringstream oss;
		oss << this->inUnsigned()->getData();

		Log::sendMessage(Log::Info, oss.str());
	}

	void PrintFloat::print()
	{
		std::ostringstream oss;
		oss << this->inFloat()->getData();

		Log::sendMessage(Log::Info, oss.str());
	}
}