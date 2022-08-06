#pragma once
#include <ghc/fs_std.hpp>

#include "Field.h"

namespace dyno {
	using FilePath = fs::path;

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
}