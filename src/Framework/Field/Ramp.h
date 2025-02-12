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

		/**
		 * @brief Get the value value"Y" of the curve by value"X" .
		 */
		float getCurveValueByX(float inputX);
		/**
		 * @brief Update the data of the Bezier curve points.
		 */
		void updateBezierCurve()override;

		double calculateLengthForPointSet(std::vector<Coord2D> BezierPtSet);
		/**
		 * @brief Resample Bezier curve.
		 */
		void updateResampleBezierCurve();
		/**
		 * @brief Reordering points on canvas boundaries.
		 */		
		void borderCloseResort();
		/**
		 * @brief Updating the data of a Field
		 */
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
				ramp->setVarByStr(substr, ramp->useBezierInterpolation);
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
				ramp->setVarByStr(substr, ramp->mNewMinY);
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