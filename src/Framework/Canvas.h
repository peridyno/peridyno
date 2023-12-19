
#pragma once
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


	};


}

