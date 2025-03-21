#include "Canvas.h"

void dyno::Canvas::addPoint(float x, float y)
{
	Coord2D a = Coord2D(x, y);
	mUserCoord.push_back(a);

	UpdateFieldFinalCoord();
}

void dyno::Canvas::addPointAndHandlePoint(Coord2D point, Coord2D handle_1, Coord2D handle_2)
{
	mUserCoord.push_back(point);
	mUserHandle.push_back(handle_1);
	mUserHandle.push_back(handle_2);

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


void dyno::Canvas::updateBezierCurve()
{
	mBezierPoint.clear();

	int n = mUserCoord.size();
	int bn = mUserHandle.size();

	if (bn != 2 * n)
	{
		rebuildHandlePoint(mUserCoord);
	}

	for (int i = 0; i < n - 1; i++)
	{
		Coord2D p0 = mUserCoord[i];
		Coord2D p1 = mUserHandle[2 * i + 1];
		Coord2D p2 = mUserHandle[2 * (i + 1)];
		Coord2D p3 = mUserCoord[i + 1];
		updateBezierPointToBezierSet(p0, p1, p2, p3, mBezierPoint);
	}
	if (mClose && n >= 3)
	{
		Coord2D p0 = mUserCoord[n - 1];
		Coord2D p1 = mUserHandle[bn - 1];
		Coord2D p2 = mUserHandle[0];
		Coord2D p3 = mUserCoord[0];
		updateBezierPointToBezierSet(p0, p1, p2, p3, mBezierPoint);
	}
	if (mClose)
	{
		if (mUserCoord.size())
		{
			mBezierPoint.push_back(mUserCoord[0]);
		}
	}
	else
	{
		if (mUserCoord.size())
		{
			mBezierPoint.push_back(mUserCoord[mUserCoord.size() - 1]);
		}
	}
}

void dyno::Canvas::updateBezierPointToBezierSet(Coord2D p0, Coord2D p1, Coord2D p2, Coord2D p3, std::vector<Coord2D>& bezierSet)
{

	Coord2D P[4] = { p0,p1,p2,p3 };

	int n = 4;
	Coord2D Pf;
	float t;
	float unit = 1.0f / 15.0f;	//bezier resolution

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
	temp.assign(mUserCoord.begin(), mUserCoord.end());
	if (mClose && temp.size() >= 3)
	{
		temp.push_back(mUserCoord[0]);
	}
	buildSegMent_Length_Map(temp);
	resamplePointFromLine(temp);

}

void dyno::Canvas::resamplePointFromLine(std::vector<Coord2D> pointSet)
{
	Coord2D P;//画点
	float uL = mSpacing / 100;
	float curL = 0;
	int n;
	if (mLength_EndPoint_Map.size())
	{
		n = (--mLength_EndPoint_Map.end())->first / uL;
	}
	double addTempLength = 0;

	mResamplePoint.clear();
	if (mUserCoord.size() >= 2)
	{
		for (size_t i = 0; i < mLengthArray.size(); i++)
		{
			double tempL = mLengthArray[i];
			auto it = mLength_EndPoint_Map.find(tempL);
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
		if (mClose)
		{
			mResamplePoint.push_back(mUserCoord[0]);
		}
		else
		{
			mResamplePoint.push_back(mUserCoord[mUserCoord.size() - 1]);
		}
	}
}

void dyno::Canvas::rebuildHandlePoint(std::vector<Coord2D> coordSet)
{
	mUserHandle.clear();
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

		mUserHandle.push_back(Coord2D(p1));
		mUserHandle.push_back(Coord2D(p2));

	}
}

void dyno::Canvas::buildSegMent_Length_Map(std::vector<Coord2D> BezierPtSet)
{
	mLength_EndPoint_Map.clear();
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
			mLength_EndPoint_Map[length] = EndPoint(e, f);

		}
	}

}

void dyno::Canvas::setInterpMode(bool useBezier)
{
	mInterpMode = useBezier ? Interpolation::Bezier : Interpolation::Linear;
}

void dyno::Canvas::clearMyCoord()
{
	mUserCoord.clear();

	mBezierPoint.clear();
	mUserHandle.clear();
}


