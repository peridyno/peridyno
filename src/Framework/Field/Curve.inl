#ifndef CURVE_SERIALIZATION
#define CURVE_SERIALIZATION

#include "Field.h"
#include "Curve.h"

namespace dyno {

	template<>
	inline std::string FVar<Curve>::serialize()
	{

		std::string finalText;
		//serialize Array
		this->getValue().convertCoordToStr("UserPoints", this->getValue().Curve::getUserPoints(), finalText);
		this->getValue().convertCoordToStr("UserHandles", this->getValue().Curve::getUserHandles(), finalText);

		//serialize Var
		this->getValue().convertVarToStr("Resample", this->getValue().getResample(), finalText);
		this->getValue().convertVarToStr("Spacing", this->getValue().getSpacing(), finalText);

		this->getValue().convertVarToStr("Close", this->getValue().getClose(), finalText);
		this->getValue().convertVarToStr("InterpMode", this->getValue().getInterpMode(), finalText);

		std::stringstream ss;
		ss << finalText;

		return ss.str();
	}


	template<>
	inline bool FVar<Curve>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;


		std::stringstream ss(str);
		std::string substr;


		auto ramp = std::make_shared<Curve>();

		int countMyCoord = -1;
		int countHandle = -1;
		int countRange = -2;
		double tempCoord = 0;
		std::string currentVarName;

		while (ss >> substr)
		{
			if (isalpha(substr[0]))
			{
				currentVarName = substr;
			}

			if (currentVarName == "UserPoints")
			{
				countMyCoord++;
				if (countMyCoord > 0 && countMyCoord % 2 != 0)
				{
					tempCoord = std::stod(substr);
				}
				else if (countMyCoord > 0 && countMyCoord % 2 == 0)
				{
					ramp->getUserPoints().push_back(Curve::Coord2D(tempCoord, std::stod(substr)));
				}
			}
			else if (currentVarName == "UserHandles")
			{
				countHandle++;
				if (countHandle > 0 && countHandle % 2 != 0)
				{
					tempCoord = std::stod(substr);
				}
				else if (countHandle > 0 && countHandle % 2 == 0)
				{
					ramp->getUserHandles().push_back(Curve::Coord2D(tempCoord, std::stod(substr)));
				}
			}
			else if (currentVarName == "Resample")
				ramp->setVarByStr(substr, ramp->getResample());
			else if (currentVarName == "Spacing")
				ramp->setVarByStr(substr, ramp->getSpacing());
			else if (currentVarName == "Close")
				ramp->setVarByStr(substr, ramp->getClose());
			else if (currentVarName == "InterpMode")
				ramp->setVarByStr(substr, ramp->getInterpMode());
		}

		ramp->updateBezierCurve();
		ramp->UpdateFieldFinalCoord();

		this->setValue(*ramp);

		return true;
	}

	template class FVar<Curve>;
}

#endif // !CURVE_SERIALIZATION