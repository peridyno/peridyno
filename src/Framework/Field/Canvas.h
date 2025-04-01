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
		void setCurveClose(bool s) { this->mClose = s; UpdateFieldFinalCoord(); };
		void setInterpMode(bool useBezier);
		void setResample(bool s) { this->mResample = s; UpdateFieldFinalCoord(); };
		virtual bool isSquard(){ return true; };
		void useBezier();
		void useLinear();
		void setSpacing(double s) { this->mSpacing = s; UpdateFieldFinalCoord(); };

		//************** Widget Interface **************//

		virtual void updateBezierCurve();
		void updateBezierPointToBezierSet(Coord2D p0, Coord2D p1, Coord2D p2, Coord2D p3, std::vector<Coord2D>& bezierSet);

		void updateResampleLinearLine();
		void resamplePointFromLine(std::vector<Coord2D> pointSet);

		//get
		std::vector<Coord2D> getPoints() { return mFinalCoord; }
		unsigned getPointSize() { return this->mFinalCoord.size(); }

		Canvas::Interpolation& getInterpMode() { return mInterpMode; }

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
			Str.append(VarName + " ");
			if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
			{
				Str.append(std::to_string(value));
			}
			else
			{
				Str.append(std::to_string(static_cast<int>(value)));
			}
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

		std::vector<Coord2D>& getUserPoints() { return mUserCoord; }
		std::vector<Coord2D>& getUserHandles() {return mUserHandle;}

		bool& getResample() { return mResample; }
		bool& getClose() { return mClose; }
		float& getSpacing() { return mSpacing; }

	protected:
		void rebuildHandlePoint(std::vector<Coord2D> coordSet);
		void buildSegMent_Length_Map(std::vector<Coord2D> BezierPtSet);




	protected:

		Canvas::Interpolation mInterpMode = Linear;
		float mSpacing = 5;
		bool mResample = true;
		bool mClose = false;
		std::vector<Coord2D> mUserCoord;
		std::vector<Coord2D> mUserHandle;

		//
		std::vector<Coord2D> mBezierPoint;
		std::vector<Coord2D> mFinalCoord;
		std::vector<Coord2D> mResamplePoint;
		std::map<float, EndPoint> mLength_EndPoint_Map;
		std::vector<double> mLengthArray;
	};

}

#endif