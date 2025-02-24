#pragma once
#include <Platform.h>

namespace dyno
{
	class Color
	{
	public:
		DYN_FUNC Color() { r = 0.0f; g = 0.0f; b = 0.0f; }

		explicit DYN_FUNC Color(float c) { r = c; g = c; b = c; }

		//explicit DYN_FUNC Color(int _r, int _g, int _b) { r = float(_r) / 255; g = float(_g) / 255; float(_b) / 255; }

		explicit DYN_FUNC Color(float _r, float _g, float _b) { r = _r; g = _g; b = _b; }

		DYN_FUNC ~Color() {};

		DYN_FUNC void HSVtoRGB(float h, float s, float v)
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

		static Color Snow() { return Color(1.0f, 0.98039f, 0.98039f); }
		static Color GhostWhite() { return Color(0.97255f, 0.97255f, 1.0f); }
		static Color WhiteSmoke() { return Color(0.96078f, 0.96078f, 0.96078f); }
		static Color Gainsboro() { return Color(0.86275f, 0.86275f, 0.86275f); }
		static Color FloralWhite() { return Color(1.0f, 0.98039f, 0.94118f); }
		static Color OldLace() { return Color(0.99216f, 0.96078f, 0.90196f); }
		static Color Linen() { return Color(0.98039f, 0.94118f, 0.90196f); }
		static Color AntiqueWhite() { return Color(0.98039f, 0.92157f, 0.84314f); }
		static Color PapayaWhip() { return Color(1.0f, 0.93725f, 0.83529f); }
		static Color BlanchedAlmond() { return Color(1.0f, 0.92157f, 0.80392f); }
		static Color Bisque() { return Color(1.0f, 0.89412f, 0.76863f); }
		static Color PeachPuff() { return Color(1.0f, 0.8549f, 0.72549f); }
		static Color NavajoWhite() { return Color(1.0f, 0.87059f, 0.67843f); }
		static Color Moccasin() { return Color(1.0f, 0.89412f, 0.7098f); }
		static Color Cornsilk() { return Color(1.0f, 0.97255f, 0.86275f); }
		static Color Ivory() { return Color(1.0f, 1.0f, 0.94118f); }
		static Color LemonChiffon() { return Color(1.0f, 0.98039f, 0.80392f); }
		static Color Seashell() { return Color(1.0f, 0.96078f, 0.93333f); }
		static Color Honeydew() { return Color(0.94118f, 1.0f, 0.94118f); }
		static Color MintCream() { return Color(0.96078f, 1.0f, 0.98039f); }
		static Color Azure() { return Color(0.94118f, 1.0f, 1.0f); }
		static Color AliceBlue() { return Color(0.94118f, 0.97255f, 1.0f); }
		static Color lavender() { return Color(0.90196f, 0.90196f, 0.98039f); }
		static Color LavenderBlush() { return Color(1.0f, 0.94118f, 0.96078f); }
		static Color MistyRose() { return Color(1.0f, 0.89412f, 0.88235f); }
		static Color White() { return Color(1.0f, 1.0f, 1.0f); }
		static Color Black() { return Color(0.0f, 0.0f, 0.0f); }
		static Color DarkSlateGray() { return Color(0.18431f, 0.3098f, 0.3098f); }
		static Color DimGrey() { return Color(0.41176f, 0.41176f, 0.41176f); }
		static Color SlateGrey() { return Color(0.43922f, 0.50196f, 0.56471f); }
		static Color LightSlateGray() { return Color(0.46667f, 0.53333f, 0.6f); }
		static Color Grey() { return Color(0.7451f, 0.7451f, 0.7451f); }
		static Color LightGray() { return Color(0.82745f, 0.82745f, 0.82745f); }
		static Color MidnightBlue() { return Color(0.09804f, 0.09804f, 0.43922f); }
		static Color NavyBlue() { return Color(0.0f, 0.0f, 0.50196f); }
		static Color CornflowerBlue() { return Color(0.39216f, 0.58431f, 0.92941f); }
		static Color DarkSlateBlue() { return Color(0.28235f, 0.23922f, 0.5451f); }
		static Color SlateBlue() { return Color(0.41569f, 0.35294f, 0.80392f); }
		static Color MediumSlateBlue() { return Color(0.48235f, 0.40784f, 0.93333f); }
		static Color LightSlateBlue() { return Color(0.51765f, 0.43922f, 1.0f); }
		static Color MediumBlue() { return Color(0.0f, 0.0f, 0.80392f); }
		static Color RoyalBlue() { return Color(0.2549f, 0.41176f, 0.88235f); }
		static Color Blue() { return Color(0.0f, 0.0f, 1.0f); }
		static Color DodgerBlue() { return Color(0.11765f, 0.56471f, 1.0f); }
		static Color DeepSkyBlue() { return Color(0.0f, 0.74902f, 1.0f); }
		static Color SkyBlue() { return Color(0.52941f, 0.80784f, 0.92157f); }
		static Color LightSkyBlue() { return Color(0.52941f, 0.80784f, 0.98039f); }
		static Color SteelBlue() { return Color(0.27451f, 0.5098f, 0.70588f); }
		static Color LightSteelBlue() { return Color(0.6902f, 0.76863f, 0.87059f); }
		static Color LightBlue() { return Color(0.67843f, 0.84706f, 0.90196f); }
		static Color PowderBlue() { return Color(0.6902f, 0.87843f, 0.90196f); }
		static Color PaleTurquoise() { return Color(0.68627f, 0.93333f, 0.93333f); }
		static Color DarkTurquoise() { return Color(0.0f, 0.80784f, 0.81961f); }
		static Color MediumTurquoise() { return Color(0.28235f, 0.81961f, 0.8f); }
		static Color Turquoise() { return Color(0.25098f, 0.87843f, 0.81569f); }
		static Color Cyan() { return Color(0.0f, 1.0f, 1.0f); }
		static Color LightCyan() { return Color(0.87843f, 1.0f, 1.0f); }
		static Color CadetBlue() { return Color(0.37255f, 0.61961f, 0.62745f); }
		static Color MediumAquamarine() { return Color(0.4f, 0.80392f, 0.66667f); }
		static Color Aquamarine() { return Color(0.49804f, 1.0f, 0.83137f); }
		static Color DarkGreen() { return Color(0.0f, 0.39216f, 0.0f); }
		static Color DarkOliveGreen() { return Color(0.33333f, 0.41961f, 0.18431f); }
		static Color DarkSeaGreen() { return Color(0.56078f, 0.73725f, 0.56078f); }
		static Color SeaGreen() { return Color(0.18039f, 0.5451f, 0.34118f); }
		static Color MediumSeaGreen() { return Color(0.23529f, 0.70196f, 0.44314f); }
		static Color LightSeaGreen() { return Color(0.12549f, 0.69804f, 0.66667f); }
		static Color PaleGreen() { return Color(0.59608f, 0.98431f, 0.59608f); }
		static Color SpringGreen() { return Color(0.0f, 1.0f, 0.49804f); }
		static Color LawnGreen() { return Color(0.48627f, 0.98824f, 0.0f); }
		static Color Green() { return Color(0.0f, 1.0f, 0.0f); }
		static Color Chartreuse() { return Color(0.49804f, 1.0f, 0.0f); }
		static Color MedSpringGreen() { return Color(0.0f, 0.98039f, 0.60392f); }
		static Color GreenYellow() { return Color(0.67843f, 1.0f, 0.18431f); }
		static Color LimeGreen() { return Color(0.19608f, 0.80392f, 0.19608f); }
		static Color YellowGreen() { return Color(0.60392f, 0.80392f, 0.19608f); }
		static Color ForestGreen() { return Color(0.13333f, 0.5451f, 0.13333f); }
		static Color OliveDrab() { return Color(0.41961f, 0.55686f, 0.13725f); }
		static Color DarkKhaki() { return Color(0.74118f, 0.71765f, 0.41961f); }
		static Color PaleGoldenrod() { return Color(0.93333f, 0.9098f, 0.66667f); }
		static Color LtGoldenrodYello() { return Color(0.98039f, 0.98039f, 0.82353f); }
		static Color LightYellow() { return Color(1.0f, 1.0f, 0.87843f); }
		static Color Yellow() { return Color(1.0f, 1.0f, 0.0f); }
		static Color Gold() { return Color(1.0f, 0.84314f, 0.0f); }
		static Color LightGoldenrod() { return Color(0.93333f, 0.86667f, 0.5098f); }
		static Color goldenrod() { return Color(0.8549f, 0.64706f, 0.12549f); }
		static Color DarkGoldenrod() { return Color(0.72157f, 0.52549f, 0.04314f); }
		static Color RosyBrown() { return Color(0.73725f, 0.56078f, 0.56078f); }
		static Color IndianRed() { return Color(0.80392f, 0.36078f, 0.36078f); }
		static Color SaddleBrown() { return Color(0.5451f, 0.27059f, 0.07451f); }
		static Color Sienna() { return Color(0.62745f, 0.32157f, 0.17647f); }
		static Color Peru() { return Color(0.80392f, 0.52157f, 0.24706f); }
		static Color Burlywood() { return Color(0.87059f, 0.72157f, 0.52941f); }
		static Color Beige() { return Color(0.96078f, 0.96078f, 0.86275f); }
		static Color Wheat() { return Color(0.96078f, 0.87059f, 0.70196f); }
		static Color SandyBrown() { return Color(0.95686f, 0.64314f, 0.37647f); }
		static Color Tan() { return Color(0.82353f, 0.70588f, 0.54902f); }
		static Color Chocolate() { return Color(0.82353f, 0.41176f, 0.11765f); }
		static Color Firebrick() { return Color(0.69804f, 0.13333f, 0.13333f); }
		static Color Brown() { return Color(0.64706f, 0.16471f, 0.16471f); }
		static Color DarkSalmon() { return Color(0.91373f, 0.58824f, 0.47843f); }
		static Color Salmon() { return Color(0.98039f, 0.50196f, 0.44706f); }
		static Color LightSalmon() { return Color(1.0f, 0.62745f, 0.47843f); }
		static Color Orange() { return Color(1.0f, 0.64706f, 0.0f); }
		static Color DarkOrange() { return Color(1.0f, 0.54902f, 0.0f); }
		static Color Coral() { return Color(1.0f, 0.49804f, 0.31373f); }
		static Color LightCoral() { return Color(0.94118f, 0.50196f, 0.50196f); }
		static Color Tomato() { return Color(1.0f, 0.38824f, 0.27843f); }
		static Color OrangeRed() { return Color(1.0f, 0.27059f, 0.0f); }
		static Color Red() { return Color(1.0f, 0.0f, 0.0f); }
		static Color HotPink() { return Color(1.0f, 0.41176f, 0.70588f); }
		static Color DeepPink() { return Color(1.0f, 0.07843f, 0.57647f); }
		static Color Pink() { return Color(1.0f, 0.75294f, 0.79608f); }
		static Color LightPink() { return Color(1.0f, 0.71373f, 0.75686f); }
		static Color PaleVioletRed() { return Color(0.85882f, 0.43922f, 0.57647f); }
		static Color Maroon() { return Color(0.6902f, 0.18824f, 0.37647f); }
		static Color MediumVioletRed() { return Color(0.78039f, 0.08235f, 0.52157f); }
		static Color VioletRed() { return Color(0.81569f, 0.12549f, 0.56471f); }
		static Color Magenta() { return Color(1.0f, 0.0f, 1.0f); }
		static Color Violet() { return Color(0.93333f, 0.5098f, 0.93333f); }
		static Color Plum() { return Color(0.86667f, 0.62745f, 0.86667f); }
		static Color Orchid() { return Color(0.8549f, 0.43922f, 0.83922f); }
		static Color MediumOrchid() { return Color(0.72941f, 0.33333f, 0.82745f); }
		static Color DarkOrchid() { return Color(0.6f, 0.19608f, 0.8f); }
		static Color DarkViolet() { return Color(0.58039f, 0.0f, 0.82745f); }
		static Color BlueViolet() { return Color(0.54118f, 0.16863f, 0.88627f); }
		static Color Purple() { return Color(0.62745f, 0.12549f, 0.94118f); }
		static Color MediumPurple() { return Color(0.57647f, 0.43922f, 0.85882f); }
		static Color Thistle() { return Color(0.84706f, 0.74902f, 0.84706f); }
		static Color Snow1() { return Color(1.0f, 0.98039f, 0.98039f); }
		static Color Snow2() { return Color(0.93333f, 0.91373f, 0.91373f); }
		static Color Snow3() { return Color(0.80392f, 0.78824f, 0.78824f); }
		static Color Snow4() { return Color(0.5451f, 0.53725f, 0.53725f); }
		static Color Seashell1() { return Color(1.0f, 0.96078f, 0.93333f); }
		static Color Seashell2() { return Color(0.93333f, 0.89804f, 0.87059f); }
		static Color Seashell3() { return Color(0.80392f, 0.77255f, 0.74902f); }
		static Color Seashell4() { return Color(0.5451f, 0.52549f, 0.5098f); }
		static Color AntiqueWhite1() { return Color(1.0f, 0.93725f, 0.85882f); }
		static Color AntiqueWhite2() { return Color(0.93333f, 0.87451f, 0.8f); }
		static Color AntiqueWhite3() { return Color(0.80392f, 0.75294f, 0.6902f); }
		static Color AntiqueWhite4() { return Color(0.5451f, 0.51373f, 0.47059f); }
		static Color Bisque1() { return Color(1.0f, 0.89412f, 0.76863f); }
		static Color Bisque2() { return Color(0.93333f, 0.83529f, 0.71765f); }
		static Color Bisque3() { return Color(0.80392f, 0.71765f, 0.61961f); }
		static Color Bisque4() { return Color(0.5451f, 0.4902f, 0.41961f); }
		static Color PeachPuff1() { return Color(1.0f, 0.8549f, 0.72549f); }
		static Color PeachPuff2() { return Color(0.93333f, 0.79608f, 0.67843f); }
		static Color PeachPuff3() { return Color(0.80392f, 0.68627f, 0.58431f); }
		static Color PeachPuff4() { return Color(0.5451f, 0.46667f, 0.39608f); }
		static Color NavajoWhite1() { return Color(1.0f, 0.87059f, 0.67843f); }
		static Color NavajoWhite2() { return Color(0.93333f, 0.81176f, 0.63137f); }
		static Color NavajoWhite3() { return Color(0.80392f, 0.70196f, 0.5451f); }
		static Color NavajoWhite4() { return Color(0.5451f, 0.47451f, 0.36863f); }
		static Color LemonChiffon1() { return Color(1.0f, 0.98039f, 0.80392f); }
		static Color LemonChiffon2() { return Color(0.93333f, 0.91373f, 0.74902f); }
		static Color LemonChiffon3() { return Color(0.80392f, 0.78824f, 0.64706f); }
		static Color LemonChiffon4() { return Color(0.5451f, 0.53725f, 0.43922f); }
		static Color Cornsilk1() { return Color(1.0f, 0.97255f, 0.86275f); }
		static Color Cornsilk2() { return Color(0.93333f, 0.9098f, 0.80392f); }
		static Color Cornsilk3() { return Color(0.80392f, 0.78431f, 0.69412f); }
		static Color Cornsilk4() { return Color(0.5451f, 0.53333f, 0.47059f); }
		static Color Ivory1() { return Color(1.0f, 1.0f, 0.94118f); }
		static Color Ivory2() { return Color(0.93333f, 0.93333f, 0.87843f); }
		static Color Ivory3() { return Color(0.80392f, 0.80392f, 0.75686f); }
		static Color Ivory4() { return Color(0.5451f, 0.5451f, 0.51373f); }
		static Color Honeydew1() { return Color(0.94118f, 1.0f, 0.94118f); }
		static Color Honeydew2() { return Color(0.87843f, 0.93333f, 0.87843f); }
		static Color Honeydew3() { return Color(0.75686f, 0.80392f, 0.75686f); }
		static Color Honeydew4() { return Color(0.51373f, 0.5451f, 0.51373f); }
		static Color LavenderBlush1() { return Color(1.0f, 0.94118f, 0.96078f); }
		static Color LavenderBlush2() { return Color(0.93333f, 0.87843f, 0.89804f); }
		static Color LavenderBlush3() { return Color(0.80392f, 0.75686f, 0.77255f); }
		static Color LavenderBlush4() { return Color(0.5451f, 0.51373f, 0.52549f); }
		static Color MistyRose1() { return Color(1.0f, 0.89412f, 0.88235f); }
		static Color MistyRose2() { return Color(0.93333f, 0.83529f, 0.82353f); }
		static Color MistyRose3() { return Color(0.80392f, 0.71765f, 0.7098f); }
		static Color MistyRose4() { return Color(0.5451f, 0.4902f, 0.48235f); }
		static Color Azure1() { return Color(0.94118f, 1.0f, 1.0f); }
		static Color Azure2() { return Color(0.87843f, 0.93333f, 0.93333f); }
		static Color Azure3() { return Color(0.75686f, 0.80392f, 0.80392f); }
		static Color Azure4() { return Color(0.51373f, 0.5451f, 0.5451f); }
		static Color SlateBlue1() { return Color(0.51373f, 0.43529f, 1.0f); }
		static Color SlateBlue2() { return Color(0.47843f, 0.40392f, 0.93333f); }
		static Color SlateBlue3() { return Color(0.41176f, 0.34902f, 0.80392f); }
		static Color SlateBlue4() { return Color(0.27843f, 0.23529f, 0.5451f); }
		static Color RoyalBlue1() { return Color(0.28235f, 0.46275f, 1.0f); }
		static Color RoyalBlue2() { return Color(0.26275f, 0.43137f, 0.93333f); }
		static Color RoyalBlue3() { return Color(0.22745f, 0.37255f, 0.80392f); }
		static Color RoyalBlue4() { return Color(0.15294f, 0.25098f, 0.5451f); }
		static Color Blue1() { return Color(0.0f, 0.0f, 1.0f); }
		static Color Blue2() { return Color(0.0f, 0.0f, 0.93333f); }
		static Color Blue3() { return Color(0.0f, 0.0f, 0.80392f); }
		static Color Blue4() { return Color(0.0f, 0.0f, 0.5451f); }
		static Color DodgerBlue1() { return Color(0.11765f, 0.56471f, 1.0f); }
		static Color DodgerBlue2() { return Color(0.1098f, 0.52549f, 0.93333f); }
		static Color DodgerBlue3() { return Color(0.09412f, 0.4549f, 0.80392f); }
		static Color DodgerBlue4() { return Color(0.06275f, 0.30588f, 0.5451f); }
		static Color SteelBlue1() { return Color(0.38824f, 0.72157f, 1.0f); }
		static Color SteelBlue2() { return Color(0.36078f, 0.67451f, 0.93333f); }
		static Color SteelBlue3() { return Color(0.3098f, 0.58039f, 0.80392f); }
		static Color SteelBlue4() { return Color(0.21176f, 0.39216f, 0.5451f); }
		static Color DeepSkyBlue1() { return Color(0.0f, 0.74902f, 1.0f); }
		static Color DeepSkyBlue2() { return Color(0.0f, 0.69804f, 0.93333f); }
		static Color DeepSkyBlue3() { return Color(0.0f, 0.60392f, 0.80392f); }
		static Color DeepSkyBlue4() { return Color(0.0f, 0.40784f, 0.5451f); }
		static Color SkyBlue1() { return Color(0.52941f, 0.80784f, 1.0f); }
		static Color SkyBlue2() { return Color(0.49412f, 0.75294f, 0.93333f); }
		static Color SkyBlue3() { return Color(0.42353f, 0.65098f, 0.80392f); }
		static Color SkyBlue4() { return Color(0.2902f, 0.43922f, 0.5451f); }
		static Color LightSkyBlue1() { return Color(0.6902f, 0.88627f, 1.0f); }
		static Color LightSkyBlue2() { return Color(0.64314f, 0.82745f, 0.93333f); }
		static Color LightSkyBlue3() { return Color(0.55294f, 0.71373f, 0.80392f); }
		static Color LightSkyBlue4() { return Color(0.37647f, 0.48235f, 0.5451f); }
		static Color SlateGray1() { return Color(0.77647f, 0.88627f, 1.0f); }
		static Color SlateGray2() { return Color(0.72549f, 0.82745f, 0.93333f); }
		static Color SlateGray3() { return Color(0.62353f, 0.71373f, 0.80392f); }
		static Color SlateGray4() { return Color(0.42353f, 0.48235f, 0.5451f); }
		static Color LightSteelBlue1() { return Color(0.79216f, 0.88235f, 1.0f); }
		static Color LightSteelBlue2() { return Color(0.73725f, 0.82353f, 0.93333f); }
		static Color LightSteelBlue3() { return Color(0.63529f, 0.7098f, 0.80392f); }
		static Color LightSteelBlue4() { return Color(0.43137f, 0.48235f, 0.5451f); }
		static Color LightBlue1() { return Color(0.74902f, 0.93725f, 1.0f); }
		static Color LightBlue2() { return Color(0.69804f, 0.87451f, 0.93333f); }
		static Color LightBlue3() { return Color(0.60392f, 0.75294f, 0.80392f); }
		static Color LightBlue4() { return Color(0.40784f, 0.51373f, 0.5451f); }
		static Color LightCyan1() { return Color(0.87843f, 1.0f, 1.0f); }
		static Color LightCyan2() { return Color(0.81961f, 0.93333f, 0.93333f); }
		static Color LightCyan3() { return Color(0.70588f, 0.80392f, 0.80392f); }
		static Color LightCyan4() { return Color(0.47843f, 0.5451f, 0.5451f); }
		static Color PaleTurquoise1() { return Color(0.73333f, 1.0f, 1.0f); }
		static Color PaleTurquoise2() { return Color(0.68235f, 0.93333f, 0.93333f); }
		static Color PaleTurquoise3() { return Color(0.58824f, 0.80392f, 0.80392f); }
		static Color PaleTurquoise4() { return Color(0.4f, 0.5451f, 0.5451f); }
		static Color CadetBlue1() { return Color(0.59608f, 0.96078f, 1.0f); }
		static Color CadetBlue2() { return Color(0.55686f, 0.89804f, 0.93333f); }
		static Color CadetBlue3() { return Color(0.47843f, 0.77255f, 0.80392f); }
		static Color CadetBlue4() { return Color(0.32549f, 0.52549f, 0.5451f); }
		static Color Turquoise1() { return Color(0.0f, 0.96078f, 1.0f); }
		static Color Turquoise2() { return Color(0.0f, 0.89804f, 0.93333f); }
		static Color Turquoise3() { return Color(0.0f, 0.77255f, 0.80392f); }
		static Color Turquoise4() { return Color(0.0f, 0.52549f, 0.5451f); }
		static Color Cyan1() { return Color(0.0f, 1.0f, 1.0f); }
		static Color Cyan2() { return Color(0.0f, 0.93333f, 0.93333f); }
		static Color Cyan3() { return Color(0.0f, 0.80392f, 0.80392f); }
		static Color Cyan4() { return Color(0.0f, 0.5451f, 0.5451f); }
		static Color DarkSlateGray1() { return Color(0.59216f, 1.0f, 1.0f); }
		static Color DarkSlateGray2() { return Color(0.55294f, 0.93333f, 0.93333f); }
		static Color DarkSlateGray3() { return Color(0.47451f, 0.80392f, 0.80392f); }
		static Color DarkSlateGray4() { return Color(0.32157f, 0.5451f, 0.5451f); }
		static Color Aquamarine1() { return Color(0.49804f, 1.0f, 0.83137f); }
		static Color Aquamarine2() { return Color(0.46275f, 0.93333f, 0.77647f); }
		static Color Aquamarine3() { return Color(0.4f, 0.80392f, 0.66667f); }
		static Color Aquamarine4() { return Color(0.27059f, 0.5451f, 0.4549f); }
		static Color DarkSeaGreen1() { return Color(0.75686f, 1.0f, 0.75686f); }
		static Color DarkSeaGreen2() { return Color(0.70588f, 0.93333f, 0.70588f); }
		static Color DarkSeaGreen3() { return Color(0.60784f, 0.80392f, 0.60784f); }
		static Color DarkSeaGreen4() { return Color(0.41176f, 0.5451f, 0.41176f); }
		static Color SeaGreen1() { return Color(0.32941f, 1.0f, 0.62353f); }
		static Color SeaGreen2() { return Color(0.30588f, 0.93333f, 0.58039f); }
		static Color SeaGreen3() { return Color(0.26275f, 0.80392f, 0.50196f); }
		static Color SeaGreen4() { return Color(0.18039f, 0.5451f, 0.34118f); }
		static Color PaleGreen1() { return Color(0.60392f, 1.0f, 0.60392f); }
		static Color PaleGreen2() { return Color(0.56471f, 0.93333f, 0.56471f); }
		static Color PaleGreen3() { return Color(0.48627f, 0.80392f, 0.48627f); }
		static Color PaleGreen4() { return Color(0.32941f, 0.5451f, 0.32941f); }
		static Color SpringGreen1() { return Color(0.0f, 1.0f, 0.49804f); }
		static Color SpringGreen2() { return Color(0.0f, 0.93333f, 0.46275f); }
		static Color SpringGreen3() { return Color(0.0f, 0.80392f, 0.4f); }
		static Color SpringGreen4() { return Color(0.0f, 0.5451f, 0.27059f); }
		static Color Green1() { return Color(0.0f, 1.0f, 0.0f); }
		static Color Green2() { return Color(0.0f, 0.93333f, 0.0f); }
		static Color Green3() { return Color(0.0f, 0.80392f, 0.0f); }
		static Color Green4() { return Color(0.0f, 0.5451f, 0.0f); }
		static Color Chartreuse1() { return Color(0.49804f, 1.0f, 0.0f); }
		static Color Chartreuse2() { return Color(0.46275f, 0.93333f, 0.0f); }
		static Color Chartreuse3() { return Color(0.4f, 0.80392f, 0.0f); }
		static Color Chartreuse4() { return Color(0.27059f, 0.5451f, 0.0f); }
		static Color OliveDrab1() { return Color(0.75294f, 1.0f, 0.24314f); }
		static Color OliveDrab2() { return Color(0.70196f, 0.93333f, 0.22745f); }
		static Color OliveDrab3() { return Color(0.60392f, 0.80392f, 0.19608f); }
		static Color OliveDrab4() { return Color(0.41176f, 0.5451f, 0.13333f); }
		static Color DarkOliveGreen1() { return Color(0.79216f, 1.0f, 0.43922f); }
		static Color DarkOliveGreen2() { return Color(0.73725f, 0.93333f, 0.40784f); }
		static Color DarkOliveGreen3() { return Color(0.63529f, 0.80392f, 0.35294f); }
		static Color DarkOliveGreen4() { return Color(0.43137f, 0.5451f, 0.23922f); }
		static Color Khaki1() { return Color(1.0f, 0.96471f, 0.56078f); }
		static Color Khaki2() { return Color(0.93333f, 0.90196f, 0.52157f); }
		static Color Khaki3() { return Color(0.80392f, 0.77647f, 0.45098f); }
		static Color Khaki4() { return Color(0.5451f, 0.52549f, 0.30588f); }
		static Color LightGoldenrod1() { return Color(1.0f, 0.92549f, 0.5451f); }
		static Color LightGoldenrod2() { return Color(0.93333f, 0.86275f, 0.5098f); }
		static Color LightGoldenrod3() { return Color(0.80392f, 0.7451f, 0.43922f); }
		static Color LightGoldenrod4() { return Color(0.5451f, 0.50588f, 0.29804f); }
		static Color LightYellow1() { return Color(1.0f, 1.0f, 0.87843f); }
		static Color LightYellow2() { return Color(0.93333f, 0.93333f, 0.81961f); }
		static Color LightYellow3() { return Color(0.80392f, 0.80392f, 0.70588f); }
		static Color LightYellow4() { return Color(0.5451f, 0.5451f, 0.47843f); }
		static Color Yellow1() { return Color(1.0f, 1.0f, 0.0f); }
		static Color Yellow2() { return Color(0.93333f, 0.93333f, 0.0f); }
		static Color Yellow3() { return Color(0.80392f, 0.80392f, 0.0f); }
		static Color Yellow4() { return Color(0.5451f, 0.5451f, 0.0f); }
		static Color Gold1() { return Color(1.0f, 0.84314f, 0.0f); }
		static Color Gold2() { return Color(0.93333f, 0.78824f, 0.0f); }
		static Color Gold3() { return Color(0.80392f, 0.67843f, 0.0f); }
		static Color Gold4() { return Color(0.5451f, 0.45882f, 0.0f); }
		static Color Goldenrod1() { return Color(1.0f, 0.75686f, 0.1451f); }
		static Color Goldenrod2() { return Color(0.93333f, 0.70588f, 0.13333f); }
		static Color Goldenrod3() { return Color(0.80392f, 0.60784f, 0.11373f); }
		static Color Goldenrod4() { return Color(0.5451f, 0.41176f, 0.07843f); }
		static Color DarkGoldenrod1() { return Color(1.0f, 0.72549f, 0.05882f); }
		static Color DarkGoldenrod2() { return Color(0.93333f, 0.67843f, 0.0549f); }
		static Color DarkGoldenrod3() { return Color(0.80392f, 0.58431f, 0.04706f); }
		static Color DarkGoldenrod4() { return Color(0.5451f, 0.39608f, 0.03137f); }
		static Color RosyBrown1() { return Color(1.0f, 0.75686f, 0.75686f); }
		static Color RosyBrown2() { return Color(0.93333f, 0.70588f, 0.70588f); }
		static Color RosyBrown3() { return Color(0.80392f, 0.60784f, 0.60784f); }
		static Color RosyBrown4() { return Color(0.5451f, 0.41176f, 0.41176f); }
		static Color IndianRed1() { return Color(1.0f, 0.41569f, 0.41569f); }
		static Color IndianRed2() { return Color(0.93333f, 0.38824f, 0.38824f); }
		static Color IndianRed3() { return Color(0.80392f, 0.33333f, 0.33333f); }
		static Color IndianRed4() { return Color(0.5451f, 0.22745f, 0.22745f); }
		static Color Sienna1() { return Color(1.0f, 0.5098f, 0.27843f); }
		static Color Sienna2() { return Color(0.93333f, 0.47451f, 0.25882f); }
		static Color Sienna3() { return Color(0.80392f, 0.40784f, 0.22353f); }
		static Color Sienna4() { return Color(0.5451f, 0.27843f, 0.14902f); }
		static Color Burlywood1() { return Color(1.0f, 0.82745f, 0.60784f); }
		static Color Burlywood2() { return Color(0.93333f, 0.77255f, 0.56863f); }
		static Color Burlywood3() { return Color(0.80392f, 0.66667f, 0.4902f); }
		static Color Burlywood4() { return Color(0.5451f, 0.45098f, 0.33333f); }
		static Color Wheat1() { return Color(1.0f, 0.90588f, 0.72941f); }
		static Color Wheat2() { return Color(0.93333f, 0.84706f, 0.68235f); }
		static Color Wheat3() { return Color(0.80392f, 0.72941f, 0.58824f); }
		static Color Wheat4() { return Color(0.5451f, 0.49412f, 0.4f); }
		static Color Tan1() { return Color(1.0f, 0.64706f, 0.3098f); }
		static Color Tan2() { return Color(0.93333f, 0.60392f, 0.28627f); }
		static Color Tan3() { return Color(0.80392f, 0.52157f, 0.24706f); }
		static Color Tan4() { return Color(0.5451f, 0.35294f, 0.16863f); }
		static Color Chocolate1() { return Color(1.0f, 0.49804f, 0.14118f); }
		static Color Chocolate2() { return Color(0.93333f, 0.46275f, 0.12941f); }
		static Color Chocolate3() { return Color(0.80392f, 0.4f, 0.11373f); }
		static Color Chocolate4() { return Color(0.5451f, 0.27059f, 0.07451f); }
		static Color Firebrick1() { return Color(1.0f, 0.18824f, 0.18824f); }
		static Color Firebrick2() { return Color(0.93333f, 0.17255f, 0.17255f); }
		static Color Firebrick3() { return Color(0.80392f, 0.14902f, 0.14902f); }
		static Color Firebrick4() { return Color(0.5451f, 0.10196f, 0.10196f); }
		static Color Brown1() { return Color(1.0f, 0.25098f, 0.25098f); }
		static Color Brown2() { return Color(0.93333f, 0.23137f, 0.23137f); }
		static Color Brown3() { return Color(0.80392f, 0.2f, 0.2f); }
		static Color Brown4() { return Color(0.5451f, 0.13725f, 0.13725f); }
		static Color Salmon1() { return Color(1.0f, 0.54902f, 0.41176f); }
		static Color Salmon2() { return Color(0.93333f, 0.5098f, 0.38431f); }
		static Color Salmon3() { return Color(0.80392f, 0.43922f, 0.32941f); }
		static Color Salmon4() { return Color(0.5451f, 0.29804f, 0.22353f); }
		static Color LightSalmon1() { return Color(1.0f, 0.62745f, 0.47843f); }
		static Color LightSalmon2() { return Color(0.93333f, 0.58431f, 0.44706f); }
		static Color LightSalmon3() { return Color(0.80392f, 0.50588f, 0.38431f); }
		static Color LightSalmon4() { return Color(0.5451f, 0.34118f, 0.25882f); }
		static Color Orange1() { return Color(1.0f, 0.64706f, 0.0f); }
		static Color Orange2() { return Color(0.93333f, 0.60392f, 0.0f); }
		static Color Orange3() { return Color(0.80392f, 0.52157f, 0.0f); }
		static Color Orange4() { return Color(0.5451f, 0.35294f, 0.0f); }
		static Color DarkOrange1() { return Color(1.0f, 0.49804f, 0.0f); }
		static Color DarkOrange2() { return Color(0.93333f, 0.46275f, 0.0f); }
		static Color DarkOrange3() { return Color(0.80392f, 0.4f, 0.0f); }
		static Color DarkOrange4() { return Color(0.5451f, 0.27059f, 0.0f); }
		static Color Coral1() { return Color(1.0f, 0.44706f, 0.33725f); }
		static Color Coral2() { return Color(0.93333f, 0.41569f, 0.31373f); }
		static Color Coral3() { return Color(0.80392f, 0.35686f, 0.27059f); }
		static Color Coral4() { return Color(0.5451f, 0.24314f, 0.18431f); }
		static Color Tomato1() { return Color(1.0f, 0.38824f, 0.27843f); }
		static Color Tomato2() { return Color(0.93333f, 0.36078f, 0.25882f); }
		static Color Tomato3() { return Color(0.80392f, 0.3098f, 0.22353f); }
		static Color Tomato4() { return Color(0.5451f, 0.21176f, 0.14902f); }
		static Color OrangeRed1() { return Color(1.0f, 0.27059f, 0.0f); }
		static Color OrangeRed2() { return Color(0.93333f, 0.25098f, 0.0f); }
		static Color OrangeRed3() { return Color(0.80392f, 0.21569f, 0.0f); }
		static Color OrangeRed4() { return Color(0.5451f, 0.1451f, 0.0f); }
		static Color Red1() { return Color(1.0f, 0.0f, 0.0f); }
		static Color Red2() { return Color(0.93333f, 0.0f, 0.0f); }
		static Color Red3() { return Color(0.80392f, 0.0f, 0.0f); }
		static Color Red4() { return Color(0.5451f, 0.0f, 0.0f); }
		static Color DeepPink1() { return Color(1.0f, 0.07843f, 0.57647f); }
		static Color DeepPink2() { return Color(0.93333f, 0.07059f, 0.53725f); }
		static Color DeepPink3() { return Color(0.80392f, 0.06275f, 0.46275f); }
		static Color DeepPink4() { return Color(0.5451f, 0.03922f, 0.31373f); }
		static Color HotPink1() { return Color(1.0f, 0.43137f, 0.70588f); }
		static Color HotPink2() { return Color(0.93333f, 0.41569f, 0.6549f); }
		static Color HotPink3() { return Color(0.80392f, 0.37647f, 0.56471f); }
		static Color HotPink4() { return Color(0.5451f, 0.22745f, 0.38431f); }
		static Color Pink1() { return Color(1.0f, 0.7098f, 0.77255f); }
		static Color Pink2() { return Color(0.93333f, 0.66275f, 0.72157f); }
		static Color Pink3() { return Color(0.80392f, 0.56863f, 0.61961f); }
		static Color Pink4() { return Color(0.5451f, 0.38824f, 0.42353f); }
		static Color LightPink1() { return Color(1.0f, 0.68235f, 0.72549f); }
		static Color LightPink2() { return Color(0.93333f, 0.63529f, 0.67843f); }
		static Color LightPink3() { return Color(0.80392f, 0.54902f, 0.58431f); }
		static Color LightPink4() { return Color(0.5451f, 0.37255f, 0.39608f); }
		static Color PaleVioletRed1() { return Color(1.0f, 0.5098f, 0.67059f); }
		static Color PaleVioletRed2() { return Color(0.93333f, 0.47451f, 0.62353f); }
		static Color PaleVioletRed3() { return Color(0.80392f, 0.40784f, 0.53725f); }
		static Color PaleVioletRed4() { return Color(0.5451f, 0.27843f, 0.36471f); }
		static Color Maroon1() { return Color(1.0f, 0.20392f, 0.70196f); }
		static Color Maroon2() { return Color(0.93333f, 0.18824f, 0.6549f); }
		static Color Maroon3() { return Color(0.80392f, 0.16078f, 0.56471f); }
		static Color Maroon4() { return Color(0.5451f, 0.1098f, 0.38431f); }
		static Color VioletRed1() { return Color(1.0f, 0.24314f, 0.58824f); }
		static Color VioletRed2() { return Color(0.93333f, 0.22745f, 0.54902f); }
		static Color VioletRed3() { return Color(0.80392f, 0.19608f, 0.47059f); }
		static Color VioletRed4() { return Color(0.5451f, 0.13333f, 0.32157f); }
		static Color Magenta1() { return Color(1.0f, 0.0f, 1.0f); }
		static Color Magenta2() { return Color(0.93333f, 0.0f, 0.93333f); }
		static Color Magenta3() { return Color(0.80392f, 0.0f, 0.80392f); }
		static Color Magenta4() { return Color(0.5451f, 0.0f, 0.5451f); }
		static Color Orchid1() { return Color(1.0f, 0.51373f, 0.98039f); }
		static Color Orchid2() { return Color(0.93333f, 0.47843f, 0.91373f); }
		static Color Orchid3() { return Color(0.80392f, 0.41176f, 0.78824f); }
		static Color Orchid4() { return Color(0.5451f, 0.27843f, 0.53725f); }
		static Color Plum1() { return Color(1.0f, 0.73333f, 1.0f); }
		static Color Plum2() { return Color(0.93333f, 0.68235f, 0.93333f); }
		static Color Plum3() { return Color(0.80392f, 0.58824f, 0.80392f); }
		static Color Plum4() { return Color(0.5451f, 0.4f, 0.5451f); }
		static Color MediumOrchid1() { return Color(0.87843f, 0.4f, 1.0f); }
		static Color MediumOrchid2() { return Color(0.81961f, 0.37255f, 0.93333f); }
		static Color MediumOrchid3() { return Color(0.70588f, 0.32157f, 0.80392f); }
		static Color MediumOrchid4() { return Color(0.47843f, 0.21569f, 0.5451f); }
		static Color DarkOrchid1() { return Color(0.74902f, 0.24314f, 1.0f); }
		static Color DarkOrchid2() { return Color(0.69804f, 0.22745f, 0.93333f); }
		static Color DarkOrchid3() { return Color(0.60392f, 0.19608f, 0.80392f); }
		static Color DarkOrchid4() { return Color(0.40784f, 0.13333f, 0.5451f); }
		static Color Purple1() { return Color(0.60784f, 0.18824f, 1.0f); }
		static Color Purple2() { return Color(0.56863f, 0.17255f, 0.93333f); }
		static Color Purple3() { return Color(0.4902f, 0.14902f, 0.80392f); }
		static Color Purple4() { return Color(0.33333f, 0.10196f, 0.5451f); }
		static Color MediumPurple1() { return Color(0.67059f, 0.5098f, 1.0f); }
		static Color MediumPurple2() { return Color(0.62353f, 0.47451f, 0.93333f); }
		static Color MediumPurple3() { return Color(0.53725f, 0.40784f, 0.80392f); }
		static Color MediumPurple4() { return Color(0.36471f, 0.27843f, 0.5451f); }
		static Color Thistle1() { return Color(1.0f, 0.88235f, 1.0f); }
		static Color Thistle2() { return Color(0.93333f, 0.82353f, 0.93333f); }
		static Color Thistle3() { return Color(0.80392f, 0.7098f, 0.80392f); }
		static Color Thistle4() { return Color(0.5451f, 0.48235f, 0.5451f); }
		static Color Grey11() { return Color(0.1098f, 0.1098f, 0.1098f); }
		static Color Grey21() { return Color(0.21176f, 0.21176f, 0.21176f); }
		static Color Grey31() { return Color(0.3098f, 0.3098f, 0.3098f); }
		static Color Grey41() { return Color(0.41176f, 0.41176f, 0.41176f); }
		static Color Grey51() { return Color(0.5098f, 0.5098f, 0.5098f); }
		static Color Grey61() { return Color(0.61176f, 0.61176f, 0.61176f); }
		static Color Grey71() { return Color(0.7098f, 0.7098f, 0.7098f); }
		static Color Grey81() { return Color(0.81176f, 0.81176f, 0.81176f); }
		static Color Grey91() { return Color(0.9098f, 0.9098f, 0.9098f); }
		static Color DarkGrey() { return Color(0.66275f, 0.66275f, 0.66275f); }
		static Color DarkBlue() { return Color(0.0f, 0.0f, 0.5451f); }
		static Color DarkCyan() { return Color(0.0f, 0.5451f, 0.5451f); }
		static Color DarkMagenta() { return Color(0.5451f, 0.0f, 0.5451f); }
		static Color DarkRed() { return Color(0.5451f, 0.0f, 0.0f); }
		static Color LightGreen() { return Color(0.56471f, 0.93333f, 0.56471f); }


		float r;
		float g;
		float b;
		
	};
}

#include "Color.inl"
