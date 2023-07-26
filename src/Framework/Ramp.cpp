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
		if (FinalCoord.size())
		{
			int l = FinalCoord.size();
			for (size_t i = 0; i < l;i ++)
			{
				xLess = (FinalCoord[i].x > inputX) ? xLess : FinalCoord[i].x;
				yLess = (FinalCoord[i].x > inputX) ? yLess : FinalCoord[i].y;

				xGreater = (FinalCoord[l - i - 1].x < inputX) ? xGreater : FinalCoord[l - i - 1].x;
				yGreater = (FinalCoord[l - i - 1].x < inputX) ? yGreater : FinalCoord[l - i - 1].y;
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


	void Ramp::addPoint(float x, float y)
	{
		Coord2D a = Coord2D(x,y);
		MyCoord.push_back(a);

		UpdateFieldFinalCoord();

	}

	void Ramp::addPointAndHandlePoint(Coord2D point, Coord2D handle_1, Coord2D handle_2)
	{

		MyCoord.push_back(point);
		myHandlePoint.push_back(handle_1);
		myHandlePoint.push_back(handle_2);

		UpdateFieldFinalCoord();
	}


	// C++ Bezier
	void Ramp::updateBezierCurve()
	{
		myBezierPoint.clear();

		int n = MyCoord.size();
		int bn = myHandlePoint.size();

		if (bn != 2 * n) 
		{
			rebuildHandlePoint(MyCoord);
		}

		for (int i = 0; i < n - 1; i++)
		{
			Coord2D p0 = MyCoord[i];
			Coord2D p1 = myHandlePoint[2 * i + 1];
			Coord2D p2 = myHandlePoint[2 * (i + 1)];
			Coord2D p3 = MyCoord[i + 1];
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint);
		}
		if (curveClose && n >= 3) 
		{
			Coord2D p0 = MyCoord[ n - 1 ];
			Coord2D p1 = myHandlePoint[bn-1];
			Coord2D p2 = myHandlePoint[0];
			Coord2D p3 = MyCoord[0];
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint);
		}
		if (curveClose) 
		{
			if (MyCoord.size())
			{
				myBezierPoint.push_back(MyCoord[0]);
			}
		}
		else 
		{
			if (MyCoord.size()) 
			{
				myBezierPoint.push_back(MyCoord[MyCoord.size() - 1]);
			}
		}
		

		if (resample) 
		{
			if (useCurve) 
			{
				updateResampleBezierCurve();
				resamplePointFromLine(myBezierPoint_H);
			}
			else 
			{
				updateResampleLinearLine();
			}

		}
	}


	void Ramp::resamplePointFromLine(std::vector<Coord2D> pointSet)
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
		
		resamplePoint.clear();
		if (MyCoord.size() >= 2) 
		{
			for (size_t i = 0; i < lengthArray.size(); i++)
			{
				double tempL = lengthArray[i];
				auto it = length_EndPoint_Map.find(tempL);
				EndPoint ep = it->second;

				while (true)
				{
					if (addTempLength >= lengthArray[i])
					{
						break;
					}
					else
					{
						double subL = lengthArray[i] - addTempLength;
						double curLineL;
						if (i > 0)
						{
							curLineL = lengthArray[i] - lengthArray[i - 1];
						}
						else
						{
							curLineL = lengthArray[0];
						}
						double per = 1 - subL / curLineL;

						P.x = per * (pointSet[ep.second].x - pointSet[ep.first].x) + pointSet[ep.first].x;
						P.y = per * (pointSet[ep.second].y - pointSet[ep.first].y) + pointSet[ep.first].y;

						resamplePoint.push_back(P);
						addTempLength += uL;
					}

				}

			}
			if (curveClose) 
			{
				resamplePoint.push_back(MyCoord[0]);
			}
			else 
			{
				resamplePoint.push_back(MyCoord[MyCoord.size() - 1]);
			}
		}
	}


	void Ramp::updateResampleBezierCurve() 
	{
		float temp = segment;
		segment = resampleResolution;
		myBezierPoint_H.clear();

		int n = MyCoord.size();
		int bn = myHandlePoint.size();

		if (bn != 2 * n)
		{
			rebuildHandlePoint(MyCoord);
		}

		for (int i = 0; i < n - 1; i++)
		{
			Coord2D p0 = MyCoord[i];
			Coord2D p1 = myHandlePoint[2 * i + 1];
			Coord2D p2 = myHandlePoint[2 * (i + 1)];
			Coord2D p3 = MyCoord[i + 1];
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint_H);
		}
		if (curveClose && n >= 3)
		{
			Coord2D p0 = MyCoord[n - 1];
			Coord2D p1 = myHandlePoint[bn - 1];
			Coord2D p2 = myHandlePoint[0];
			Coord2D p3 = MyCoord[0];
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint_H);
		}
		if (curveClose)
		{
			if (MyCoord.size()) 
			{
				myBezierPoint_H.push_back(MyCoord[0]);
			}
		}
		else
		{
			if (MyCoord.size())
			{
				myBezierPoint_H.push_back(MyCoord[MyCoord.size() - 1]);
			}
		}

		buildSegMent_Length_Map(myBezierPoint_H);

		segment = temp;
	}

	void Ramp::updateResampleLinearLine()
	{
		std::vector<Coord2D> temp;
		temp.assign(MyCoord.begin(),MyCoord.end());
		if (curveClose) 
		{
			temp.push_back(MyCoord[0]);
		}
		buildSegMent_Length_Map(temp);
		resamplePointFromLine(temp);

	}


	void Ramp::rebuildHandlePoint(std::vector<Coord2D> coordSet)
	{
		myHandlePoint.clear();
		int ptnum = coordSet.size();
		for (size_t i = 0;i < ptnum; i++ )
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

	//bezier point
	void Ramp::updateBezierPointToBezierSet(Coord2D p0, Coord2D p1, Coord2D p2, Coord2D p3, std::vector<Coord2D>& bezierSet)
	{
		Coord2D P[4] = { p0,p1,p2,p3 };
		Coord2D Pt[4] = { p0,p1,p2,p3 };

		int n = 4;
		Coord2D Pf;
		Coord2D lastP = p0;
		double t;
		float unit = 1 / segment;

		for (t = 0; t < 1 ; t += unit)
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

	double Ramp::calculateLengthForPointSet(std::vector<Coord2D> BezierPtSet)
	{
		double length = 0;
		int n = BezierPtSet.size();
		for (size_t k = 0; k < n - 1; k++)
		{	
			int e = k ;
			int f = k + 1;
			length += sqrt(std::pow((BezierPtSet[f].x - BezierPtSet[e].x), 2) + std::pow((BezierPtSet[f].y - BezierPtSet[e].y), 2));
		}
		return length;
	}

	void Ramp::buildSegMent_Length_Map(std::vector<Coord2D> BezierPtSet)
	{
		length_EndPoint_Map.clear();
		lengthArray.clear();
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
				lengthArray.push_back(length);
				length_EndPoint_Map[length] = EndPoint(e, f);

			}
		}

	}


	//widget to field;
	void Ramp::addItemMyCoord(float x, float y)
	{
		Coord2D s;
		s.set(x, y);
		MyCoord.push_back(s);
	}

	void Ramp::addFloatItemToCoord(float x, float y, std::vector<Coord2D>& coordArray)
	{
		Coord2D s;
		s.set(x, y);
		coordArray.push_back(s);
	}

	void Ramp::addItemOriginalCoord(int x, int y) 
	{
		OriginalCoord s;
		s.set(x, y);
		Originalcoord.push_back(s);
	}

	void Ramp::addItemHandlePoint(int x, int y)
	{
		OriginalCoord s;
		s.set(x, y);
		OriginalHandlePoint.push_back(s);
	}

	void Ramp::clearMyCoord()
	{
		MyCoord.clear();
		Originalcoord.clear();
		OriginalHandlePoint.clear();
		myBezierPoint.clear();
		myHandlePoint.clear();
	}

	void Ramp::setCurveClose(bool s)
	{
		this->curveClose = s;

		UpdateFieldFinalCoord();
	}

	void Ramp::useBezier() 
	{
		setInterpMode(true);
	}

	void Ramp::useLinear()
	{
		setInterpMode(false);
	}

	void Ramp::setDisplayUseRamp(bool v) 
	{
		this->displayUseRamp = v;
	}

	void Ramp::setUseRamp(bool v)
	{
		this->useRamp = v;
	}

	void Ramp::setInterpMode(bool useBezier)
	{
		useCurve = useBezier;
		if (useBezier) 
		{
			this->InterpMode = Ramp::Interpolation::Bezier;
		}
		else 
		{
			this->InterpMode = Ramp::Interpolation::Linear;
		}	
	}

	void Ramp::setResample(bool s) 
	{
		this->resample = s;
		UpdateFieldFinalCoord();
	}

	void Ramp::setSpacing(double s) 
	{
		this->Spacing = s;
		UpdateFieldFinalCoord();
	}


	void Ramp::setUseSquard(bool s) 
	{
		useSquard = s;
	}

	void Ramp::borderCloseResort() 
	{

		std::vector<Coord2D> tempHandle;
		tempHandle.assign(myHandlePoint.begin(), myHandlePoint.end());
		FE_HandleCoord.clear();
		FE_MyCoord.clear();

		Coord2D Cfirst(-1, -1, -1);
		Coord2D Cend(-1, -1, -1);
		Coord2D F1(-1, -1, -1);
		Coord2D F2(-1, -1, -1);
		Coord2D E1(-1, -1, -1);
		Coord2D E2(-1, -1, -1);

		for (int i = 0; i < MyCoord.size(); i++)
		{
			if (MyCoord[i].x == 0)
			{
				Cfirst = MyCoord[i];
				F1 = tempHandle[2 * i];
				F2 = tempHandle[2 * i + 1];
			}
			else if (MyCoord[i].x == 1)
			{
				Cend = MyCoord[i];
				E1 = tempHandle[2 * i];
				E2 = tempHandle[2 * i + 1];
			}
			else
			{
				FE_MyCoord.push_back(MyCoord[i]);
				FE_HandleCoord.push_back(tempHandle[2 * i]);
				FE_HandleCoord.push_back(tempHandle[2 * i + 1]);
			}
		}


		if (Cend.x != -1)
		{
			FE_MyCoord.insert(FE_MyCoord.begin(), Cend);
			FE_HandleCoord.insert(FE_HandleCoord.begin(), E2);
			FE_HandleCoord.insert(FE_HandleCoord.begin(), E1);
		}
		else
		{
			FE_MyCoord.insert(FE_MyCoord.begin(), Coord2D(1, 0.5));
			FE_HandleCoord.insert(FE_HandleCoord.begin(), Coord2D(1, 0.5));
			FE_HandleCoord.insert(FE_HandleCoord.begin(), Coord2D(0.9, 0.5));
		}


		if (Cfirst.x != -1)
		{
			FE_MyCoord.insert(FE_MyCoord.begin(), Cfirst);
			FE_HandleCoord.insert(FE_HandleCoord.begin(), F2);
			FE_HandleCoord.insert(FE_HandleCoord.begin(), F1);
		}
		else
		{
			FE_MyCoord.insert(FE_MyCoord.begin(), Coord2D(0, 0.5));
			FE_HandleCoord.insert(FE_HandleCoord.begin(), Coord2D(0.1, 0.5));
			FE_HandleCoord.insert(FE_HandleCoord.begin(), Coord2D(0, 0.5));
		}


	}

	void Ramp::UpdateFieldFinalCoord()
	{
		if (this->Bordermode == BorderMode::Close) 
		{
			borderCloseResort();
		}

		FinalCoord.clear();

		if (useCurve )
		{
			if (resample) 
			{
				if (MyCoord.size() >= 2)
				{
					updateBezierCurve();
				}
				FinalCoord.assign(resamplePoint.begin(),resamplePoint.end());
			}
			else 
			{
				FinalCoord.assign(myBezierPoint.begin(), myBezierPoint.end());
			}
		}
		else if ( !useCurve ) 
		{

			if (resample) 
			{
				if (MyCoord.size() >= 2)
				{

					updateResampleLinearLine();
				}
				FinalCoord.assign(resamplePoint.begin(), resamplePoint.end());
			}
			else
			{
				FinalCoord.assign(MyCoord.begin(), MyCoord.end());
			}
		}
		

		for (size_t i = 0; i < FinalCoord.size(); i++)
		{
			FinalCoord[i].x = (NmaxX - NminX) * FinalCoord[i].x + NminX;
			FinalCoord[i].y = (NmaxY - NminY) * FinalCoord[i].y + NminY;
		}


	}







}
