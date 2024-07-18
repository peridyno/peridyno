#ifndef RAMP_H
#define RAMP_H

#include <vector>
#include <memory>
#include <string>
#include "Vector/Vector2D.h"
#include "Vector/Vector3D.h"

#include "Field.h"
#include "Canvas.h"

namespace dyno {


	class Ramp : public Canvas
	{
	public:
		Ramp();
		Ramp(Direction dir);
		Ramp(const Ramp& ramp);


		~Ramp() { ; };

		//Ramp
		float getCurveValueByX(float inputX);

		void updateBezierCurve();
		double calculateLengthForPointSet(std::vector<Coord2D> BezierPtSet);



		void updateResampleBezierCurve();


		void setRange_MinX(float min, float max) { remapRange[0] = min; remapRange[1] = max; }// "MinX", "MinY", "MaxX", "MaxY"
		void setRange_MaxX(float min, float max) { remapRange[4] = min; remapRange[5] = max; }
		void setRange_MinY(float min, float max) { remapRange[2] = min; remapRange[3] = max; }
		void setRange_MaxY(float min, float max) { remapRange[6] = min; remapRange[7] = max; }
		void setRange(float min, float max) { setRange_MinX(min, max); setRange_MaxX(min, max); setRange_MinY(min, max); setRange_MaxY(min, max); };

		void borderCloseResort();


		/*void convertCoordToStr(std::string VarName, std::vector<Ramp::Coord2D> Array, std::string& Str)
		{
			Str.append(VarName + " ");
			for (int i = 0; i < Array.size(); i++)
			{
				std::string tempTextX = std::to_string(Array[i].x);
				std::string tempTextY = std::to_string(Array[i].y);
				Str.append(tempTextX + " " + tempTextY);
				if (i != Array.size() - 1)
				{
					Str.append(" ");
				}
			}
			Str.append(" ");
		}

		template <typename T>
		void convertVarToStr(std::string VarName, T value, std::string& Str)
		{
			int temp = int(value);
			Str.append(VarName + " ");
			Str.append(std::to_string(temp));
			Str.append(" ");
			std::cout << std::endl << Str;
		}

		template<>
		void convertVarToStr(std::string VarName, float value, std::string& Str)
		{
			Str.append(VarName + " ");
			Str.append(std::to_string(value));
			Str.append(" ");
			std::cout << std::endl << Str;
		}

		template<>
		void convertVarToStr(std::string VarName, double value, std::string& Str)
		{
			Str.append(VarName + " ");
			Str.append(std::to_string(value));
			Str.append(" ");
			std::cout << std::endl << Str;
		}


		void setVarByStr(std::string Str, double& value)
		{
			if (std::isdigit(Str[0]) | (Str[0] == '-'))
			{
				value = std::stod(Str);
			}
			return;
		}
		void setVarByStr(std::string Str, float& value)
		{
			if (std::isdigit(Str[0]) | (Str[0] == '-'))
			{
				value = float(std::stod(Str));
			}
			return;
		}
		void setVarByStr(std::string Str, int& value)
		{
			if (std::isdigit(Str[0]) | (Str[0] == '-'))
			{
				value = std::stoi(Str);
			}
			return;
		}

		void setVarByStr(std::string Str, bool& value)
		{
			if (std::isdigit(Str[0]))
			{
				value = bool(std::stoi(Str));
			}
			return;
		}

		void setVarByStr(std::string Str, Interpolation& value)
		{
			if (std::isdigit(Str[0]))
			{
				value = Interpolation(std::stoi(Str));
			}
			return;
		}*/






		void UpdateFieldFinalCoord() override;
	public:
		//ave
		Direction Dirmode = x;

		std::string DirectionStrings[int(Direction::count)] = { "x","y" };
		std::vector<Coord2D> myBezierPoint_H;//

		std::vector<Coord2D> FE_MyCoord;
		std::vector<Coord2D> FE_HandleCoord;


	private:
		float xLess = 1;
		float xGreater = 0;
		float yLess = 1;
		float yGreater = 0;


	};

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
					ramp->mCoord.push_back(Ramp::Coord2D(tempCoord, std::stod(substr)));
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
				ramp->setVarByStr(substr, ramp->mInterpMode);
			else if (currentVarName == "Dirmode")
				ramp->setVarByStr(substr, ramp->Dirmode);
			else if (currentVarName == "lockSize")
				ramp->setVarByStr(substr, ramp->lockSize);
			else if (currentVarName == "useColseButton")
				ramp->setVarByStr(substr, ramp->useColseButton);
			else if (currentVarName == "useSquardButton")
				ramp->setVarByStr(substr, ramp->useSquardButton);
			else if (currentVarName == "segment")
				ramp->setVarByStr(substr, ramp->segment);
			else if (currentVarName == "resampleResolution")
				ramp->setVarByStr(substr, ramp->resampleResolution);
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

	template class FVar<Ramp>;
}

#endif // !RAMP_H