#include "Field.h"
#include "Ramp.h"


namespace dyno 
{
	template<>
	std::string FVar<bool>::serialize()
	{
		if (isEmpty())
			return "";

		bool b = this->getValue();
		return b ? "true" : "false";
	}

	template<>
	bool FVar<bool>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		bool b = str == std::string("true") ? true : false;
		this->setValue(b);

		return true;
	}

	template<>
	std::string FVar<int>::serialize()
	{
		if (isEmpty())
			return "";

		int val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	bool FVar<int>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		int val = std::stoi(str);
		this->setValue(val);

		return true;
	}

	template<>
	std::string FVar<uint>::serialize()
	{
		if (isEmpty())
			return "";

		uint val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	bool FVar<uint>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		uint val = std::stoi(str);
		this->setValue(val);

		return true;
	}

	template<>
	std::string FVar<float>::serialize()
	{
		if (isEmpty())
			return "";

		float val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	bool FVar<float>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		float val = std::stof(str);
		this->setValue(val);

		return true;
	}

	template<>
	std::string FVar<double>::serialize()
	{
		if (isEmpty())
			return "";

		double val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	bool FVar<double>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		double val = std::stod(str);
		this->setValue(val);

		return true;
	}

	template<>
	std::string FVar<Vec3f>::serialize()
	{
		if (isEmpty())
			return "";

		Vec3f val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z;

		return ss.str();
	}

	template<>
	bool FVar<Vec3f>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;
		
		ss >> substr;
		float x = std::stof(substr);

		ss >> substr;
		float y = std::stof(substr);

		ss >> substr;
		float z = std::stof(substr);


		this->setValue(Vec3f(x, y, z));

		return true;
	}

	template<>
	std::string FVar<Vec3i>::serialize()
	{
		if (isEmpty())
			return "";

		Vec3i val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z;

		return ss.str();
	}

	template<>
	bool FVar<Vec3i>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		int x = std::stoi(substr);

		ss >> substr;
		int y = std::stoi(substr);

		ss >> substr;
		int z = std::stoi(substr.c_str());

		this->setValue(Vec3i(x, y, z));

		return true;
	}

	template<>
	std::string FVar<Vec3d>::serialize()
	{
		if (isEmpty())
			return "";

		Vec3d val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z;

		return ss.str();
	}

	template<>
	bool FVar<Vec3d>::deserialize(const std::string& str)
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

		this->setValue(Vec3d(x, y, z));

		return true;
	}

// 	template<>
// 	std::string FVar<FilePath>::serialize()
// 	{
// 		if (isEmpty())
// 			return "";
// 
// 		FilePath val = this->getValue();
// 
// 		return val.string();
// 	}
// 
// 	template<>
// 	bool FVar<FilePath>::deserialize(const std::string& str)
// 	{
// 		if (str.empty())
// 			return false;
// 
// 		this->setValue(str);
// 
// 		return true;
// 	}

	template<>
	std::string FVar<std::string>::serialize()
	{
		if (isEmpty())
			return "";

		std::string val = this->getValue();

		return val;
	}

	template<>
	bool FVar<std::string>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		this->setValue(str);

		return true;
	}

	template<>
	std::string FVar<Ramp>:: serialize()
	{

		std::string finalText;
		//serialize Array
		this->getValue().convertCoordToStr("MyCoord", this->getValue().Ramp::MyCoord, finalText);
		this->getValue().convertCoordToStr("myHandlePoint", this->getValue().Ramp::myHandlePoint, finalText);

		//serialize Var
		this->getValue().convertVarToStr("useCurve", this->getValue().useCurve, finalText);
		this->getValue().convertVarToStr("resample", this->getValue().resample, finalText);
		this->getValue().convertVarToStr("useSquard", this->getValue().useSquard, finalText);
		this->getValue().convertVarToStr("Spacing", this->getValue().Spacing, finalText);
		this->getValue().convertVarToStr("displayUseRamp", this->getValue().displayUseRamp, finalText);
		this->getValue().convertVarToStr("useRamp", this->getValue().useRamp, finalText);
		this->getValue().convertVarToStr("NminX", this->getValue().NminX, finalText);
		this->getValue().convertVarToStr("NmaxX", this->getValue().NmaxX, finalText);
		this->getValue().convertVarToStr("NminY", this->getValue().NminY, finalText);
		this->getValue().convertVarToStr("NmaxY", this->getValue().NmaxY, finalText);

		this->getValue().convertVarToStr("curveClose", this->getValue().curveClose, finalText);
		this->getValue().convertVarToStr("InterpMode", this->getValue().InterpMode, finalText);
		//this->getValue().convertVarToStr("Bordermode", this->getValue().Bordermode, finalText);

		//
		this->getValue().convertVarToStr("Dirmode", this->getValue().Dirmode, finalText);
		this->getValue().convertVarToStr("lockSize", this->getValue().lockSize, finalText);
		this->getValue().convertVarToStr("useColseButton", this->getValue().useColseButton, finalText);
		this->getValue().convertVarToStr("useSquardButton", this->getValue().useSquardButton, finalText);
		this->getValue().convertVarToStr("handleDefaultLength", this->getValue().handleDefaultLength, finalText);
		this->getValue().convertVarToStr("segment", this->getValue().segment, finalText);
		this->getValue().convertVarToStr("resampleResolution", this->getValue().resampleResolution, finalText);
		this->getValue().convertVarToStr("displayUseRamp", this->getValue().displayUseRamp, finalText);
		this->getValue().convertVarToStr("useRamp", this->getValue().useRamp, finalText);
		
		{
			finalText.append("remapRange");
			finalText.append(" ");
			for (int i = 0; i < 8; i++) 
			{
				finalText.append(std::to_string(this->getValue().remapRange[i]));
				if (i != 7) { finalText.append(" "); }
			}
			finalText.append(" ");
		}

		std::cout << std::endl << finalText;

		std::stringstream ss;
		ss << finalText;

		return ss.str();
	}
	template<>
	bool FVar<Ramp>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;


		std::stringstream ss(str);
		std::string substr;


		auto ramp = std::make_shared<Ramp>();

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

			if (currentVarName == "MyCoord")
			{
				countMyCoord++;
				if (countMyCoord > 0 && countMyCoord % 2 != 0)
				{
					tempCoord = std::stod(substr);
				}
				else if (countMyCoord > 0 && countMyCoord % 2 == 0)
				{
					ramp->MyCoord.push_back(Ramp::Coord2D(tempCoord, std::stod(substr)));
				}
			}
			else if (currentVarName == "myHandlePoint")
			{
				countHandle++;
				if (countHandle > 0 && countHandle % 2 != 0)
				{
					tempCoord = std::stod(substr);
				}
				else if (countHandle > 0 && countHandle % 2 == 0)
				{
					ramp->myHandlePoint.push_back(Ramp::Coord2D(tempCoord, std::stod(substr)));
				}
			}
			else if (currentVarName == "useCurve")
				ramp->setVarByStr(substr, ramp->useCurve);
			else if (currentVarName == "resample")
				ramp->setVarByStr(substr, ramp->resample);
			else if (currentVarName == "useSquard")
				ramp->setVarByStr(substr, ramp->useSquard);
			else if (currentVarName == "Spacing")
				ramp->setVarByStr(substr, ramp->Spacing);
			else if (currentVarName == "displayUseRamp")
				ramp->setVarByStr(substr, ramp->displayUseRamp);
			else if (currentVarName == "useRamp")
				ramp->setVarByStr(substr, ramp->useRamp);
			else if (currentVarName == "NminX")
				ramp->setVarByStr(substr, ramp->NminX);
			else if (currentVarName == "NmaxX")
				ramp->setVarByStr(substr, ramp->NmaxX);
			else if (currentVarName == "NminY")
				ramp->setVarByStr(substr, ramp->NminY);
			else if (currentVarName == "NmaxY")
				ramp->setVarByStr(substr, ramp->NmaxY);
			else if (currentVarName == "curveClose")
				ramp->setVarByStr(substr, ramp->curveClose);
			else if (currentVarName == "InterpMode")
				ramp->setVarByStr(substr, ramp->InterpMode);
			//else if (currentVarName == "Bordermode")
			//	ramp->setVarByStr(substr, ramp->Bordermode);
			else if (currentVarName == "Dirmode")
				ramp->setVarByStr(substr, ramp->Dirmode);
			else if (currentVarName == "lockSize")
				ramp->setVarByStr(substr, ramp->lockSize);
			else if (currentVarName == "useColseButton")
				ramp->setVarByStr(substr, ramp->useColseButton);
			else if (currentVarName == "useSquardButton")
				ramp->setVarByStr(substr, ramp->useSquardButton);
			else if (currentVarName == "handleDefaultLength")
				ramp->setVarByStr(substr, ramp->handleDefaultLength);
			else if (currentVarName == "segment")
				ramp->setVarByStr(substr, ramp->segment);
			else if (currentVarName == "resampleResolution")
				ramp->setVarByStr(substr, ramp->resampleResolution);
			else if (currentVarName == "displayUseRamp")
				ramp->setVarByStr(substr, ramp->displayUseRamp);
			else if (currentVarName == "useSquardButton")
				ramp->setVarByStr(substr, ramp->useSquardButton);
			else if (currentVarName == "useSquardButton")
				ramp->setVarByStr(substr, ramp->useSquardButton);
			else if (currentVarName == "remapRange")
			{
				countRange++;
				if (countRange >= 0)
				{
					ramp->remapRange[countRange] = std::stof(substr);
				}
			}
			
		}



		ramp->updateBezierCurve();
		ramp->UpdateFieldFinalCoord();

		this->setValue(*ramp);



		return true;
	}



	template class FVar<bool>;
	template class FVar<int>;
	template class FVar<uint>;
	template class FVar<float>;
	template class FVar<double>;
	template class FVar<Vec3f>;
	template class FVar<Vec3d>;
	template class FVar<Vec3i>;
	template class FVar<std::string>;
	template class FVar<Ramp>;

}
