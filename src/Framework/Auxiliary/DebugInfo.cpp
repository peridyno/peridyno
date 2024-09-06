#include "DebugInfo.h"
#include "Log.h"

#include <sstream>

namespace dyno 
{
	void DebugInfo::updateImpl()
	{
		this->print();
	}

	PrintInt::PrintInt()
	{
		this->varForceUpdate()->setValue(true);
	}

	void PrintInt::print()
	{
		std::ostringstream oss;
		oss << this->inInt()->getData();

		Log::sendMessage(Log::Info, oss.str());
	}

	PrintUnsigned::PrintUnsigned()
	{
		this->varForceUpdate()->setValue(true);
	}

	void PrintUnsigned::print()
	{
		std::ostringstream oss;
		oss << this->inUnsigned()->getData();

		Log::sendMessage(Log::Info, oss.str());
	}

	PrintFloat::PrintFloat()
	{
		this->varForceUpdate()->setValue(true);
	}

	void PrintFloat::print()
	{
		std::ostringstream oss;
		oss << this->inFloat()->getData();

		Log::sendMessage(Log::Info, oss.str());
	}

	PrintVector::PrintVector()
	{
		this->varForceUpdate()->setValue(true);
	}

	void PrintVector::print()
	{
		std::ostringstream oss;
		auto vec = this->inVector()->getData();
		oss << vec.x << " " << vec.y << " " << vec.z;

		Log::sendMessage(Log::Info, oss.str());
	}
}