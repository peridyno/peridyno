
#pragma once
#include <vector>
#include <memory>
#include <string>
#include "Field.h"

namespace dyno {
	class Ramp
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
		enum BorderMode
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
			Coord2D(double a,double b,int i) 
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
			EndPoint(int first,int second) 
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
		struct idHandlePoint
		{
			int id = 0;
			int x1 = 0;
			int y1 = 0;
			int x2 = 0;
			int y2 = 0;

			void set(int id,int x1, int y1,int x2,int y2)
			{
				this->id = id;
				this->x1 = x1;
				this->y1 = y1;
				this->x2 = x2;
				this->y2 = y2;
			}
		};

	public:
		Ramp();
		Ramp(BorderMode border) ;
		Ramp(const Ramp & ramp) 
		{
			this->Dirmode = ramp.Dirmode;
			this->Bordermode = ramp.Bordermode;
			this->MyCoord = ramp.MyCoord;
			this->FE_MyCoord = ramp.FE_MyCoord;
			this->FE_HandleCoord = ramp.FE_HandleCoord;
			this->FinalCoord = ramp.FinalCoord;
			this->Originalcoord = ramp.Originalcoord;
			this->OriginalHandlePoint = ramp.OriginalHandlePoint;

			this->myBezierPoint = ramp.myBezierPoint;
			this->myBezierPoint_H = ramp.myBezierPoint_H;
			this->resamplePoint = ramp.resamplePoint;
			this->myHandlePoint = ramp.myHandlePoint;

			this->lengthArray = ramp.lengthArray;
			this->length_EndPoint_Map = ramp.length_EndPoint_Map;

			this->InterpMode = ramp.InterpMode;

			this->remapRange[8] = ramp.InterpMode;// "MinX","MinY","MaxX","MaxY"

			this->lockSize = ramp.lockSize;
			this->useCurve = ramp.useCurve;

			this->useSquard = ramp.useSquard;
			this->curveClose = ramp.curveClose;
			this->resample = ramp.resample;

			this->useColseButton = ramp.useColseButton;
			this->useSquardButton = ramp.useSquardButton;

			this->Spacing = ramp.Spacing;

			this->NminX = ramp.NminX;
			this->NmaxX = ramp.NmaxX;
			this->NminY = ramp.NminY;
			this->NmaxY = ramp.NmaxY;

			this->handleDefaultLength = ramp.handleDefaultLength;
			this->segment = ramp.segment;
			this->resampleResolution = ramp.resampleResolution;

			this->xLess = ramp.xLess;
			this->xGreater = ramp.xGreater;
			this->yLess = ramp.yLess;
			this->yGreater = ramp.yGreater;

			this->generatorMin = ramp.generatorMin;
			this->generatorMax = ramp.generatorMax;

			this->customHandle = ramp.customHandle;

		}

		~Ramp() { ; };

		void varChanged();
		float getCurveValueByX(float inputX);
		void addItemMyCoord(float x ,float y);
		void addItemBezierCoord(float x, float y);
		void addFloatItemToCoord(float x, float y,std::vector<Coord2D>& coordArray);
		void addItemOriginalCoord(int x, int y);
		void addPoint(float x, float y);
		void setCurveClose(bool s);
		void addPointAndHandlePoint(Coord2D point,Coord2D handle_1, Coord2D handle_2 );
		void clearMyCoord();
		void addItemHandlePoint(int x, int y);
		void UpdateFieldFinalCoord();
		void updateBezierPointToBezierSet(Coord2D p0, Coord2D p1, Coord2D p2, Coord2D p3 ,std::vector<Coord2D>& bezierSet);
		void updateBezierCurve();
		void rebuildHandlePoint(std::vector<Coord2D> s);
		double calculateLengthForPointSet(std::vector<Coord2D> BezierPtSet);
		void setInterpMode(bool useBezier);
		void setUseSquard(bool s);
		void useBezier();
		void useLinear();
		void setResample(bool s);
		void remapX(double minX, double maxX) { NminX = minX; NmaxX = maxX; UpdateFieldFinalCoord();}
		void remapY(double minY, double maxY) { NminY = minY; NmaxY = maxY; UpdateFieldFinalCoord();}
		void remapXY(double minX, double maxX, double minY, double maxY) { NminX = minX; NmaxX = maxX; NminY = minY; NmaxY = maxY; UpdateFieldFinalCoord();}

		void buildSegMent_Length_Map(std::vector<Coord2D> BezierPtSet);
		void updateResampleLinearLine();
		void updateResampleBezierCurve();
		void resamplePointFromLine(std::vector<Coord2D> pointSet);

		void setRange_MinX(float min, float max) { remapRange[0] = min; remapRange[1] = max; }// "MinX", "MinY", "MaxX", "MaxY"
		void setRange_MaxX(float min, float max) { remapRange[4] = min; remapRange[5] = max; }
		void setRange_MinY(float min, float max) { remapRange[2] = min; remapRange[3] = max; }
		void setRange_MaxY(float min, float max) { remapRange[6] = min; remapRange[7] = max; }
		void setRange(float min, float max) { setRange_MinX(min, max); setRange_MaxX(min, max); setRange_MinY(min, max); setRange_MaxY(min, max); };
		void setBorderMode(BorderMode border) { this->Bordermode = border; UpdateFieldFinalCoord();}
		void setSpacing(double s);
		void borderCloseResort();
		void setDisplayUseRamp(bool v);
		void setUseRamp(bool v);


		Direction Dirmode = x;
		std::string DirectionStrings[int(Direction::count)] = { "x","y" };
		BorderMode Bordermode = Close;
		std::vector<Coord2D> MyCoord;
		std::vector<Coord2D> FE_MyCoord;
		std::vector<Coord2D> FE_HandleCoord;
		std::vector<Coord2D> FinalCoord;
		std::vector<OriginalCoord> Originalcoord;//qt Point Coord
		std::vector<OriginalCoord> OriginalHandlePoint;//qt HandlePoint Coord

		std::vector<Coord2D> myBezierPoint;
		std::vector<Coord2D> myBezierPoint_H;
		std::vector<Coord2D> resamplePoint;
		std::vector<Coord2D> myHandlePoint;

		std::vector<double> lengthArray;
		std::map<float, EndPoint> length_EndPoint_Map;

		std::string InterpStrings[int(Direction::count)] = { "Linear","Bezier" };
		Interpolation InterpMode = Linear;

		float remapRange[8] = {-3,3,-3,3,-3,3,-3,3};// "MinX","MinY","MaxX","MaxY"

		bool lockSize = false;
		bool useCurve = false;


		bool resample = true;

		bool curveClose = false;
		bool useColseButton = true;

		bool useSquard = false;
		bool useSquardButton = true;

		Real Spacing = 5;

		double NminX = 0;
		double NmaxX = 1;
		double NminY = 0;
		double NmaxY = 1;

		int handleDefaultLength = 18;
		float segment = 10;
		float resampleResolution = 20;


	private:
		float xLess = 1;
		float xGreater = 0;
		float yLess = 1;
		float yGreater = 0;

		bool generatorMin = true;
		bool generatorMax = true;

		bool customHandle = false;

	};

	


}

