#ifndef COLOR_SERIALIZATION
#define COLOR_SERIALIZATION

#include "Field.h"

namespace dyno
{
	template<>
	std::string FVar<Color>::serialize() override
	{
		if (isEmpty())
			return "";

		Color val = this->getValue();

		std::stringstream ss;
		ss << val.r << " " << val.g << " " << val.b;

		return ss.str();
	}

	template<>
	bool FVar<Color>::deserialize(const std::string& str) override
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		double x = std::stod(substr);

		ss >> substr;
		double y = std::stod(substr);

		ss >> substr;
		double z = std::stod(substr);

		this->setValue(Color(x, y, z));

		return true;
	}

	template class FVar<Color>;
}

#endif // !COLOR_SERIALIZATION
