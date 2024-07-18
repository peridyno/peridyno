#ifndef CURVE_H
#define CURVE_H

#pragma once
#include <vector>
#include <memory>
#include <string>
#include "Vector/Vector2D.h"
#include "Vector/Vector3D.h"

#include "Field.h"
#include "Canvas.h"

namespace dyno {

	class Curve :public Canvas
	{
	public:

		Curve();
		Curve(CurveMode mode) 
		{
			curveClose = int(mode);
		}
		Curve(const Curve& curve);


		~Curve() { };

	public:



	private:



	public:
	
		//interface:

		//Qt:

		void updateResampleBezierCurve(std::vector<Coord2D>& myBezierPoint_H);


		//Remapping Coord
		
		//IO
		
		void UpdateFieldFinalCoord() override;

		

	private:




	};





	template<>
	std::string FVar<Curve>::serialize()
	{

		std::string finalText;
		//serialize Array
		this->getValue().convertCoordToStr("MyCoord", this->getValue().Curve::mCoord, finalText);
		this->getValue().convertCoordToStr("myHandlePoint", this->getValue().Curve::myHandlePoint, finalText);

		//serialize Var
		this->getValue().convertVarToStr("useCurve", this->getValue().useCurve, finalText);
		this->getValue().convertVarToStr("resample", this->getValue().resample, finalText);
		this->getValue().convertVarToStr("useSquard", this->getValue().useSquard, finalText);
		this->getValue().convertVarToStr("Spacing", this->getValue().Spacing, finalText);

		this->getValue().convertVarToStr("NminX", this->getValue().NminX, finalText);
		this->getValue().convertVarToStr("NmaxX", this->getValue().NmaxX, finalText);
		this->getValue().convertVarToStr("NminY", this->getValue().NminY, finalText);
		this->getValue().convertVarToStr("NmaxY", this->getValue().NmaxY, finalText);

		this->getValue().convertVarToStr("curveClose", this->getValue().curveClose, finalText);
		this->getValue().convertVarToStr("InterpMode", this->getValue().mInterpMode, finalText);
		this->getValue().convertVarToStr("lockSize", this->getValue().lockSize, finalText);
		this->getValue().convertVarToStr("useColseButton", this->getValue().useColseButton, finalText);
		this->getValue().convertVarToStr("useSquardButton", this->getValue().useSquardButton, finalText);


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
	bool FVar<Curve>::deserialize(const std::string& str)
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

			if (currentVarName == "MyCoord")
			{
				countMyCoord++;
				if (countMyCoord > 0 && countMyCoord % 2 != 0)
				{
					tempCoord = std::stod(substr);
				}
				else if (countMyCoord > 0 && countMyCoord % 2 == 0)
				{
					ramp->mCoord.push_back(Curve::Coord2D(tempCoord, std::stod(substr)));
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
					ramp->myHandlePoint.push_back(Curve::Coord2D(tempCoord, std::stod(substr)));
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
			else if (currentVarName == "lockSize")
				ramp->setVarByStr(substr, ramp->lockSize);
			else if (currentVarName == "useColseButton")
				ramp->setVarByStr(substr, ramp->useColseButton);
			else if (currentVarName == "useSquardButton")
				ramp->setVarByStr(substr, ramp->useSquardButton);
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

	template class FVar<Curve>;
}

#endif