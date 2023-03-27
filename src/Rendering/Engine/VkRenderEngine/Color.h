#pragma once
#include <math.h>
#include <iostream>
#include "Platform.h"

namespace px
{
	class Color
	{
	public:
		Color();
		Color(float _r, float _g, float _b) { r = _r; g = _g; b = _b; }

		~Color() {};

		void HSVtoRGB(float h, float s, float v)
		{
			int i;
			float f, p, q, t;

			if (s == 0) {
				// achromatic (grey)
				r = g = b = v;
				return;
			}

			h /= 60;       // sector 0 to 5
			i = (int)floor(h);
			f = h - i;        // factorial part of h
			p = v * (1 - s);
			q = v * (1 - s * f);
			t = v * (1 - s * (1 - f));

			switch (i) {
			case 0:
				r = v;
				g = t;
				b = p;
				break;
			case 1:
				r = q;
				g = v;
				b = p;
				break;
			case 2:
				r = p;
				g = v;
				b = t;
				break;
			case 3:
				r = p;
				g = q;
				b = v;
				break;
			case 4:
				r = t;
				g = p;
				b = v;
				break;
			default:    // case 5:
				r = v;
				g = p;
				b = q;
				break;
			}
		}

		float r;
		float g;
		float b;
	};
}