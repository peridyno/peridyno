
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
		enum BorderMode
		{
			Open = 0,
			Close = 1,
		};
		struct MyCoord2D
		{
			float x = 0;
			float y = 0;

			void set(float a, float b) 
			{
				this->x = a;
				this->y = b;
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
		Ramp();
		Ramp(BorderMode border) ;
		~Ramp() { ; };
		void varChanged();
		float getCurveValueByX(float inputX);
		void printsize();
		void addItemMyCoord(float x ,float y);
		void addItemOriginalCoord(int x, int y);

		void clearMyCoord();
		void setOriginalCoord(int x0, int x1, int y0, int y1) { oMinX = x0; oMaxX = x1; oMinY = y0; oMaxY = y1; }

		Direction mode = x;
		std::string DirectionStrings[int(Direction::count)] = { "x","y" };
		BorderMode Bordermode = Close;
		std::vector<MyCoord2D> MyCoord;
		std::vector<OriginalCoord> Originalcoord;
		int oMaxX;
		int oMinX;
		int oMaxY;
		int oMinY;


	private:
		float xLess = 1;
		float xGreater = 0;
		float yLess = 1;
		float yGreater = 0;
	};

	


}

//template<typename Real>
//struct Ramp
//{
//	Real s = 1.0f;
//};
