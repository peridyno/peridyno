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

		void varChanged();
		float getCurveValueByX(float inputX);
		void addItemMyCoord(float x, float y);
		void addItemBezierCoord(float x, float y);
		void addFloatItemToCoord(float x, float y, std::vector<Coord2D>& coordArray);
		void addItemOriginalCoord(int x, int y);
		void addPoint(float x, float y);
		void setCurveClose(bool s);
		void addPointAndHandlePoint(Coord2D point, Coord2D handle_1, Coord2D handle_2);
		void clearMyCoord();
		void addItemHandlePoint(int x, int y);
		void UpdateFieldFinalCoord();
		void updateBezierPointToBezierSet(Coord2D p0, Coord2D p1, Coord2D p2, Coord2D p3, std::vector<Coord2D>& bezierSet);
		void updateBezierCurve();
		void rebuildHandlePoint(std::vector<Coord2D> s);
		double calculateLengthForPointSet(std::vector<Coord2D> BezierPtSet);
		void setInterpMode(bool useBezier);
		void setUseSquard(bool s);
		void useBezier();
		void useLinear();
		void setResample(bool s);
		void remapX(double minX, double maxX) { NminX = minX; NmaxX = maxX; UpdateFieldFinalCoord(); }
		void remapY(double minY, double maxY) { NminY = minY; NmaxY = maxY; UpdateFieldFinalCoord(); }
		void remapXY(double minX, double maxX, double minY, double maxY) { NminX = minX; NmaxX = maxX; NminY = minY; NmaxY = maxY; UpdateFieldFinalCoord(); }

		void buildSegMent_Length_Map(std::vector<Coord2D> BezierPtSet);
		void updateResampleLinearLine();
		void updateResampleBezierCurve();
		void resamplePointFromLine(std::vector<Coord2D> pointSet);

		void setRange_MinX(float min, float max) { remapRange[0] = min; remapRange[1] = max; }// "MinX", "MinY", "MaxX", "MaxY"
		void setRange_MaxX(float min, float max) { remapRange[4] = min; remapRange[5] = max; }
		void setRange_MinY(float min, float max) { remapRange[2] = min; remapRange[3] = max; }
		void setRange_MaxY(float min, float max) { remapRange[6] = min; remapRange[7] = max; }
		void setRange(float min, float max) { setRange_MinX(min, max); setRange_MaxX(min, max); setRange_MinY(min, max); setRange_MaxY(min, max); };
		//void setBorderMode(BorderMode border) { this->Bordermode = border; UpdateFieldFinalCoord();}
		void setSpacing(double s);
		void borderCloseResort();
		void setDisplayUseRamp(bool v);
		void setUseRamp(bool v);

		void convertCoordToStr(std::string VarName, std::vector<Ramp::Coord2D> Array, std::string& Str)
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

		void setVarByStr(std::string Str, Direction& value)
		{
			if (std::isdigit(Str[0]))
			{
				value = Direction(std::stoi(Str));
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
		}

		//must save
		Direction Dirmode = x;
		//BorderMode Bordermode = Close;
		Interpolation InterpMode = Linear;

		std::vector<Coord2D> MyCoord;
		std::vector<Coord2D> myHandlePoint;

		//
		std::string InterpStrings[int(Direction::count)] = { "Linear","Bezier" };
		std::string DirectionStrings[int(Direction::count)] = { "x","y" };

		std::vector<Coord2D> myBezierPoint;
		std::vector<Coord2D> myBezierPoint_H;
		std::vector<Coord2D> resamplePoint;

		std::vector<OriginalCoord> Originalcoord;//qt Point Coord
		std::vector<OriginalCoord> OriginalHandlePoint;//qt HandlePoint Coord

		std::vector<Coord2D> FE_MyCoord;
		std::vector<Coord2D> FE_HandleCoord;
		std::vector<Coord2D> FinalCoord;

		std::vector<double> lengthArray;
		std::map<float, EndPoint> length_EndPoint_Map;

		//must save
		float remapRange[8] = { -3,3,-3,3,-3,3,-3,3 };// "MinX","MinY","MaxX","MaxY"

		bool lockSize = false;
		bool useCurve = false;


		bool resample = true;

		bool curveClose = false;
		bool useColseButton = true;

		bool useSquard = false;
		bool useSquardButton = false;

		float Spacing = 5;

		double NminX = 0;
		double NmaxX = 1;
		double NminY = 0;
		double NmaxY = 1;

		int handleDefaultLength = 18;
		float segment = 10;
		float resampleResolution = 20;


		bool displayUseRamp = false;   // Display UseRamp checkbox for RampWidget;
		bool useRamp = false;    // Enable curve data ,enabled when displayEnable is true

	private:
		float xLess = 1;
		float xGreater = 0;
		float yLess = 1;
		float yGreater = 0;

		bool generatorMin = true;
		bool generatorMax = true;

		bool customHandle = false;

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

	template class FVar<Ramp>;
}

#endif // !RAMP_H