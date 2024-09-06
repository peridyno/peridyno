#include "FilePath.h"
#include "Field.h"

namespace dyno {
	template<>
	std::string FVar<FilePath>::serialize()
	{
		if (isEmpty())
			return "";

		FilePath val = this->getValue();

		return val.string();
	}

	template<>
	bool FVar<FilePath>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		this->setValue(str);

		return true;
	}

	template class FVar<FilePath>;
}
