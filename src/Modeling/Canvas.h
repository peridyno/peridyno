#ifndef CANVAS_H
#define CANVAS_H

#include <vector>
#include <memory>
#include <string>
#include "Vector/Vector2D.h"
#include "Vector/Vector3D.h"

namespace dyno {
	class Canvas
	{
	public:

		enum Direction
		{
			x = 0,
			y = 1,
			count = 2,
		};

		enum Interpolation
		{
			Linear = 0,
			Bezier = 1,
			InterpolationCount = 2,
		};

		enum CurveMode
		{
			Open = 0,
			Close = 1,
		};

		struct Coord2D
		{
			double x = 0;
			double y = 0;

			void set(double a, double b)
			{
				this->x = a;
				this->y = b;
			}
			Coord2D()
			{
			}
			Coord2D(double a, double b)
			{
				double temp = a;
				if (temp < 0) { temp = 0; }
				else if (temp > 1) { temp = 1; }
				this->x = temp;
				temp = b;
				if (temp < 0) { temp = 0; }
				else if (temp > 1) { temp = 1; }
				this->y = temp;
			}
			Coord2D(Vec2f s)
			{
				this->x = s[0];
				this->y = s[1];
			}
			Coord2D(double a, double b, int i)
			{
				this->x = a;
				this->y = b;
			}

		};

		struct EndPoint
		{
			int first = 0;
			int second = 0;
			EndPoint() {  }
			EndPoint(int first, int second)
			{
				this->first = first;
				this->second = second;
			}
		};

		struct OriginalCoord
		{
			int x = 0;
			int y = 0;

			void set(int a, int b)
			{
				this->x = a;
				this->y = b;
			}
		};

	public:
		Canvas() {}
		~Canvas() {}

		//User Interface
		//Edit 
		void addPoint(float x, float y);
		void addPointAndHandlePoint(Coord2D point, Coord2D handle_1, Coord2D handle_2);
		void addFloatItemToCoord(float x, float y, std::vector<Coord2D>& coordArray);
		void clearMyCoord();

		//Comands
		void setCurveClose(bool s) { this->curveClose = s; UpdateFieldFinalCoord(); };
		void setInterpMode(bool useBezier);
		void setResample(bool s) { this->resample = s; UpdateFieldFinalCoord(); };
		void setUseSquard(bool s) { useSquard = s; };
		void useBezier();
		void useLinear();
		void setSpacing(double s) { this->Spacing = s; UpdateFieldFinalCoord(); };

		//
		void remapX(double minX, double maxX) { NminX = minX; NmaxX = maxX; UpdateFieldFinalCoord(); }
		void remapY(double minY, double maxY) { mNewMinY = minY; NmaxY = maxY; UpdateFieldFinalCoord(); }
		void remapXY(double minX, double maxX, double minY, double maxY) { NminX = minX; NmaxX = maxX; mNewMinY = minY; NmaxY = maxY; UpdateFieldFinalCoord(); }
		
		void setRange_MinX(float min, float max) { remapRange[0] = min; remapRange[1] = max; }// "MinX", "MinY", "MaxX", "MaxY"
		void setRange_MaxX(float min, float max) { remapRange[4] = min; remapRange[5] = max; }
		void setRange_MinY(float min, float max) { remapRange[2] = min; remapRange[3] = max; }
		void setRange_MaxY(float min, float max) { remapRange[6] = min; remapRange[7] = max; }
		void setRange(float min, float max) { setRange_MinX(min, max); setRange_MaxX(min, max); setRange_MinY(min, max); setRange_MaxY(min, max); };



		//************** Widget Interface **************//
		void addItemOriginalCoord(int x, int y);
		void addItemHandlePoint(int x, int y);
		virtual void updateBezierCurve();
		void updateBezierPointToBezierSet(Coord2D p0, Coord2D p1, Coord2D p2, Coord2D p3, std::vector<Coord2D>& bezierSet);

		void updateResampleLinearLine();
		void resamplePointFromLine(std::vector<Coord2D> pointSet);

		//get
		std::vector<Coord2D> getPoints() { return mFinalCoord; }
		unsigned getPointSize() { return this->mFinalCoord.size(); }


		virtual void UpdateFieldFinalCoord() {};


		//IO

		void convertCoordToStr(std::string VarName, std::vector<Canvas::Coord2D> Array, std::string& Str)
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

		void setVarByStr(std::string Str, Canvas::Interpolation& value)
		{
			if (std::isdigit(Str[0]))
			{
				value = Canvas::Interpolation(std::stoi(Str));
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

	protected:
		void rebuildHandlePoint(std::vector<Coord2D> coordSet);
		void buildSegMent_Length_Map(std::vector<Coord2D> BezierPtSet);

		//IO
		


	public:
		Canvas::Interpolation mInterpMode = Linear;
		std::vector<Coord2D> mCoord;
		std::vector<Coord2D> mBezierPoint;
		std::vector<Coord2D> mFinalCoord;
		std::vector<Coord2D> mResamplePoint;
		std::vector<double> mLengthArray;
		std::vector<Coord2D> myHandlePoint;
		std::string InterpStrings[2] = { "Linear","Bezier" };
		std::vector<OriginalCoord> Originalcoord;//qt Point Coord
		std::vector<OriginalCoord> OriginalHandlePoint;//qt HandlePoint Coord

		float remapRange[8] = { -3,3,-3,3,-3,3,-3,3 };// "MinX","MinY","MaxX","MaxY"

		double NminX = 0;
		double NmaxX = 1;
		double mNewMinY = 0;
		double NmaxY = 1;

		bool lockSize = false;
		bool useBezierInterpolation = false;
		bool resample = true;
		bool curveClose = false;
		bool useColseButton = true;
		bool useSquard = true;
		bool useSquardButton = true;
		float Spacing = 5;
		float segment = 10;
		float resampleResolution = 20;

		std::map<float, EndPoint> length_EndPoint_Map;



	protected:


	};

	

}

#endif