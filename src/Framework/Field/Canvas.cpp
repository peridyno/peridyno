#include "Canvas.h"

void dyno::Canvas::addPoint(float x, float y)
{
	Coord2D a = Coord2D(x, y);
	mCoord.push_back(a);

	UpdateFieldFinalCoord();
}

void dyno::Canvas::addPointAndHandlePoint(Coord2D point, Coord2D handle_1, Coord2D handle_2)
{
	mCoord.push_back(point);
	myHandlePoint.push_back(handle_1);
	myHandlePoint.push_back(handle_2);

	UpdateFieldFinalCoord();
}

void dyno::Canvas::addFloatItemToCoord(float x, float y, std::vector<Coord2D>& coordArray)
{
	Coord2D s;
	s.set(x, y);
	coordArray.push_back(s);
}

void dyno::Canvas::useBezier()
{
	setInterpMode(true);
}

void dyno::Canvas::useLinear()
{
	setInterpMode(false);
}

void dyno::Canvas::addItemOriginalCoord(int x, int y)
{
	OriginalCoord s;
	s.set(x, y);
	Originalcoord.push_back(s);
}

void dyno::Canvas::addItemHandlePoint(int x, int y)
{
	OriginalCoord s;
	s.set(x, y);
	OriginalHandlePoint.push_back(s);
}

void dyno::Canvas::updateBezierCurve()
{
	mBezierPoint.clear();

	int n = mCoord.size();
	int bn = myHandlePoint.size();

	if (bn != 2 * n)
	{
		rebuildHandlePoint(mCoord);
	}

	for (int i = 0; i < n - 1; i++)
	{
		Coord2D p0 = mCoord[i];
		Coord2D p1 = myHandlePoint[2 * i + 1];
		Coord2D p2 = myHandlePoint[2 * (i + 1)];
		Coord2D p3 = mCoord[i + 1];
		updateBezierPointToBezierSet(p0, p1, p2, p3, mBezierPoint);
	}
	if (curveClose && n >= 3)
	{
		Coord2D p0 = mCoord[n - 1];
		Coord2D p1 = myHandlePoint[bn - 1];
		Coord2D p2 = myHandlePoint[0];
		Coord2D p3 = mCoord[0];
		updateBezierPointToBezierSet(p0, p1, p2, p3, mBezierPoint);
	}
	if (curveClose)
	{
		if (mCoord.size())
		{
			mBezierPoint.push_back(mCoord[0]);
		}
	}
	else
	{
		if (mCoord.size())
		{
			mBezierPoint.push_back(mCoord[mCoord.size() - 1]);
		}
	}
}

void dyno::Canvas::updateBezierPointToBezierSet(Coord2D p0, Coord2D p1, Coord2D p2, Coord2D p3, std::vector<Coord2D>& bezierSet)
{

	Coord2D P[4] = { p0,p1,p2,p3 };

	int n = 4;
	Coord2D Pf;
	double t;
	double unit = 1 / segment;

	for (t = 0; t < 1; t += unit)
	{
		P[0] = p0;
		P[1] = p1;
		P[2] = p2;
		P[3] = p3;

		int x = n;
		while (1)
		{
			if (x == 1)
				break;
			for (int i = 0; i < x - 1; i++)
			{
				Pf.x = (P[i + 1].x - P[i].x) * t + P[i].x;
				Pf.y = (P[i + 1].y - P[i].y) * t + P[i].y;
				P[i] = Pf;

			}
			x--;
		}
		bezierSet.push_back(Pf);
	}
}

void dyno::Canvas::updateResampleLinearLine()
{
	std::vector<Coord2D> temp;
	temp.assign(mCoord.begin(), mCoord.end());
	if (curveClose && temp.size() >= 3)
	{
		temp.push_back(mCoord[0]);
	}
	buildSegMent_Length_Map(temp);
	resamplePointFromLine(temp);

}

void dyno::Canvas::resamplePointFromLine(std::vector<Coord2D> pointSet)
{
	Coord2D P;//画点
	float uL = Spacing / 100;
	float curL = 0;
	int n;
	if (length_EndPoint_Map.size())
	{
		n = (--length_EndPoint_Map.end())->first / uL;
	}
	double addTempLength = 0;

	mResamplePoint.clear();
	if (mCoord.size() >= 2)
	{
		for (size_t i = 0; i < mLengthArray.size(); i++)
		{
			double tempL = mLengthArray[i];
			auto it = length_EndPoint_Map.find(tempL);
			EndPoint ep = it->second;

			while (true)
			{
				if (addTempLength >= mLengthArray[i])
				{
					break;
				}
				else
				{
					double subL = mLengthArray[i] - addTempLength;
					double curLineL;
					if (i > 0)
					{
						curLineL = mLengthArray[i] - mLengthArray[i - 1];
					}
					else
					{
						curLineL = mLengthArray[0];
					}
					double per = 1 - subL / curLineL;

					P.x = per * (pointSet[ep.second].x - pointSet[ep.first].x) + pointSet[ep.first].x;
					P.y = per * (pointSet[ep.second].y - pointSet[ep.first].y) + pointSet[ep.first].y;

					mResamplePoint.push_back(P);
					addTempLength += uL;
				}

			}

		}
		if (curveClose)
		{
			mResamplePoint.push_back(mCoord[0]);
		}
		else
		{
			mResamplePoint.push_back(mCoord[mCoord.size() - 1]);
		}
	}
}

void dyno::Canvas::rebuildHandlePoint(std::vector<Coord2D> coordSet)
{
	myHandlePoint.clear();
	int ptnum = coordSet.size();
	for (size_t i = 0; i < ptnum; i++)
	{
		dyno::Vec2f P(coordSet[i].x, coordSet[i].y);
		dyno::Vec2f p1;
		dyno::Vec2f p2;
		dyno::Vec2f N;

		int id = i;
		int f;
		int s;
		if (ptnum < 2)
		{
			N = Vec2f(1, 0);
		}
		else
		{
			f = id - 1;
			s = id + 1;
			if (id == 0)//首点
			{
				N[0] = coordSet[s].x - coordSet[id].x;
				N[1] = coordSet[s].y - coordSet[id].y;
			}
			else if (id == ptnum - 1)//末点
			{
				N[0] = coordSet[id].x - coordSet[f].x;
				N[1] = coordSet[id].y - coordSet[f].y;
			}
			else//中间点
			{
				N[0] = coordSet[s].x - coordSet[f].x;
				N[1] = coordSet[s].y - coordSet[f].y;
			}
		}

		N.normalize();
		double length = 0.05;
		p1 = P - N * length;
		p2 = P + N * length;

		myHandlePoint.push_back(Coord2D(p1));
		myHandlePoint.push_back(Coord2D(p2));

	}
}

void dyno::Canvas::buildSegMent_Length_Map(std::vector<Coord2D> BezierPtSet)
{
	length_EndPoint_Map.clear();
	mLengthArray.clear();
	std::vector<Coord2D> temp;

	double length = 0;

	int n = BezierPtSet.size();
	if (BezierPtSet.size())
	{
		for (size_t k = 0; k < n - 1; k++)
		{
			int e = k;
			int f = k + 1;
			length += sqrt(std::pow((BezierPtSet[f].x - BezierPtSet[e].x), 2) + std::pow((BezierPtSet[f].y - BezierPtSet[e].y), 2));
			mLengthArray.push_back(length);
			length_EndPoint_Map[length] = EndPoint(e, f);

		}
	}

}

void dyno::Canvas::setInterpMode(bool useBezier)
{
	useBezierInterpolation = useBezier;
	if (useBezier)
	{
		this->mInterpMode = Interpolation::Bezier;
	}
	else
	{
		this->mInterpMode = Interpolation::Linear;
	}
}

void dyno::Canvas::clearMyCoord()
{
	mCoord.clear();
	Originalcoord.clear();
	OriginalHandlePoint.clear();
	mBezierPoint.clear();
	myHandlePoint.clear();
}

