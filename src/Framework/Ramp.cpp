#include "Ramp.h"
#include "Field.h"

namespace dyno {
	
	template<>
	std::string FVar<Ramp>::serialize()
	{
		if (isEmpty())
			return "";

		Ramp val = this->getValue();

		return "";
	}

	template<>
	bool FVar<Ramp>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;
		return true;
	}

	Ramp::Ramp() 
	{

	}

	Ramp::Ramp(BorderMode border) 
	{ 
		Bordermode = border;
	}

	float Ramp::getCurveValueByX(float inputX)
	{
		if (MyCoord.size()) 
		{
			int l =MyCoord.size();
			for (size_t i = 0; i < l;i ++)
			{
				xLess = (MyCoord[i].x > inputX) ? xLess : MyCoord[i].x;
				yLess = (MyCoord[i].x > inputX) ? yLess : MyCoord[i].y;

				xGreater = (MyCoord[l - i - 1].x < inputX) ? xGreater : MyCoord[l - i - 1].x;
				yGreater = (MyCoord[l - i - 1].x < inputX) ? yGreater : MyCoord[l - i - 1].y;
			}
			if (xGreater !=xLess) 
			{
				float pr = (inputX - xLess) / (xGreater - xLess);
				float f = pr * (yGreater - yLess) + yLess;

				return f;
			}
			else 
			{
				return yGreater;
			}
		}
		return -1;
	}
	void Ramp::printsize()
	{
		printf("float数组大小%d\n",MyCoord.size());
		printf("original数组大小%d\n", MyCoord.size());

	};

	void Ramp::addItemMyCoord(float x, float y) 
	{
		MyCoord2D s;
		s.set(x,y);
		MyCoord.push_back(s);
		printsize();
		std::cout << s.x << "floatarray" << s.y << std::endl;
	}

	void Ramp::addItemOriginalCoord(int x, int y) 
	{
		OriginalCoord s;
		s.set(x, y);
		Originalcoord.push_back(s);
		printsize();
		std::cout << s.x << "originalarray" << s.y << std::endl;	
	}

	void Ramp::clearMyCoord()
	{
		MyCoord.clear();
		Originalcoord.clear();
		printf("clear\n");
	}



}
