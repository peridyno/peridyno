#include "Curve.h"
#include "Vector/Vector3D.h"
#include "Vector/Vector2D.h"

namespace dyno {

	Curve::Curve()
	{

	}



	Curve::Curve(const Curve& ramp)
	{
		//this->Dirmode = ramp.Dirmode;
		//this->Bordermode = ramp.Bordermode;
		this->MyCoord = ramp.MyCoord;
		this->FE_MyCoord = ramp.FE_MyCoord;
		this->FE_HandleCoord = ramp.FE_HandleCoord;
		this->FinalCoord = ramp.FinalCoord;
		this->Originalcoord = ramp.Originalcoord;
		this->OriginalHandlePoint = ramp.OriginalHandlePoint;

		this->myBezierPoint = ramp.myBezierPoint;
		this->resamplePoint = ramp.resamplePoint;
		this->myHandlePoint = ramp.myHandlePoint;

		this->lengthArray = ramp.lengthArray;
		this->length_EndPoint_Map = ramp.length_EndPoint_Map;

		this->InterpMode = ramp.InterpMode;

		this->remapRange[8] = ramp.InterpMode;// "MinX","MinY","MaxX","MaxY"

		this->lockSize = ramp.lockSize;
		this->useCurve = ramp.useCurve;
		this->displayUseRamp = ramp.displayUseRamp;
		this->useRamp = ramp.useRamp;
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

	// Update FinalCoord
	void Curve::UpdateFieldFinalCoord()
	{

		FinalCoord.clear();

		//如果使用贝塞尔插值
		if (InterpMode== Interpolation::Bezier )
		{
			if (MyCoord.size() >= 2)
			{
				updateBezierCurve();
			}
			if (resample)
			{
				std::vector<Coord2D> myBezierPoint_H;
				updateResampleBezierCurve(myBezierPoint_H);
				resamplePointFromLine(myBezierPoint_H);
				
				FinalCoord.assign(resamplePoint.begin(), resamplePoint.end());
			}

		}
		//如果使用线性插值
		else if (InterpMode == Interpolation::Linear )
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

	//C++ Edit
	void Curve::addPoint(float x, float y)
	{
		Coord2D a = Coord2D(x,y);
		MyCoord.push_back(a);

		UpdateFieldFinalCoord();

	}

	void Curve::addPointAndHandlePoint(Coord2D point, Coord2D handle_1, Coord2D handle_2)
	{

		MyCoord.push_back(point);
		myHandlePoint.push_back(handle_1);
		myHandlePoint.push_back(handle_2);

		UpdateFieldFinalCoord();
	}


	// C++ Bezier 插值Bezier曲线
	void Curve::updateBezierCurve()
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

	}


	void Curve::resamplePointFromLine(std::vector<Coord2D> pointSet)
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


	void Curve::updateResampleBezierCurve(std::vector<Coord2D>& myBezierPoint_H)
	{
		float temp = segment;
		segment = resampleResolution;
		myBezierPoint_H.clear();

		int n = MyCoord.size();
		int bn = myHandlePoint.size();

		//如果手柄数目不对，重新构建贝塞尔手柄
		if (bn != 2 * n)
		{
			rebuildHandlePoint(MyCoord);
		}

		//遍历每个线段，按照“F点、F点手柄2、E点手柄1、E点”  插值贝塞尔曲线点
		for (int i = 0; i < n - 1; i++)
		{
			Coord2D p0 = MyCoord[i];
			Coord2D p1 = myHandlePoint[2 * i + 1];
			Coord2D p2 = myHandlePoint[2 * (i + 1)];
			Coord2D p3 = MyCoord[i + 1];
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint_H);
		}
		//如果曲线闭合，且能形成完整环形。则将第一个点、最后一个点按照上述逻辑继续   插值贝塞尔曲线点
		if (curveClose && n >= 3)
		{
			Coord2D p0 = MyCoord[n - 1];
			Coord2D p1 = myHandlePoint[bn - 1];
			Coord2D p2 = myHandlePoint[0];
			Coord2D p3 = MyCoord[0];
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint_H);
		}
		//如果曲线闭合，？？？？？？
		if (curveClose)
		{
			if (MyCoord.size()) 
			{
				myBezierPoint_H.push_back(MyCoord[0]);
			}
		}
		//如果曲线不闭合，将最后一个点返回
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

	void Curve::updateResampleLinearLine()
	{
		std::vector<Coord2D> temp;
		temp.assign(MyCoord.begin(),MyCoord.end());
		if (curveClose&&temp.size()>= 3) 
		{
			temp.push_back(MyCoord[0]);
		}
		buildSegMent_Length_Map(temp);
		resamplePointFromLine(temp);

	}


	void Curve::rebuildHandlePoint(std::vector<Coord2D> coordSet)
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
	void Curve::updateBezierPointToBezierSet(Coord2D p0, Coord2D p1, Coord2D p2, Coord2D p3, std::vector<Coord2D>& bezierSet)
	{
		Coord2D P[4] = { p0,p1,p2,p3 };

		int n = 4;
		Coord2D Pf;
		double t;
		double unit = 1 / segment;

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

	double Curve::calculateLengthForPointSet(std::vector<Coord2D> BezierPtSet)
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

	void Curve::buildSegMent_Length_Map(std::vector<Coord2D> BezierPtSet)
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
	void Curve::addItemMyCoord(float x, float y)
	{
		Coord2D s;
		s.set(x, y);
		MyCoord.push_back(s);
	}

	void Curve::addFloatItemToCoord(float x, float y, std::vector<Coord2D>& coordArray)
	{
		Coord2D s;
		s.set(x, y);
		coordArray.push_back(s);
	}

	void Curve::addItemOriginalCoord(int x, int y)
	{
		OriginalCoord s;
		s.set(x, y);
		Originalcoord.push_back(s);
	}

	void Curve::addItemHandlePoint(int x, int y)
	{
		OriginalCoord s;
		s.set(x, y);
		OriginalHandlePoint.push_back(s);
	}

	void Curve::clearMyCoord()
	{
		MyCoord.clear();
		Originalcoord.clear();
		OriginalHandlePoint.clear();
		myBezierPoint.clear();
		myHandlePoint.clear();
	}

	void Curve::setCurveClose(bool s)
	{
		this->curveClose = s;

		UpdateFieldFinalCoord();
	}

	void Curve::useBezier()
	{
		setInterpMode(true);
	}

	void Curve::useLinear()
	{
		setInterpMode(false);
	}


	void Curve::setInterpMode(bool useBezier)
	{
		useCurve = useBezier;
		if (useBezier) 
		{
			this->InterpMode = Interpolation::Bezier;
		}
		else 
		{
			this->InterpMode = Interpolation::Linear;
		}	
	}

	void Curve::setResample(bool s)
	{
		this->resample = s;
		UpdateFieldFinalCoord();
	}

	void Curve::setSpacing(double s)
	{
		this->Spacing = s;
		UpdateFieldFinalCoord();
	}


	void Curve::setUseSquard(bool s)
	{
		useSquard = s;
	}



	//template<typename TDataType>
	//std::shared_ptr<PointSet<DataType3f>> getPoints()
	//{
	//	int pointSize = this->getPointSize();
	//	PointSet<TDataType> mPointSet;
	//	Coord Location;
	//	for (size_t i = 0; i < pointSize; i++)
	//	{
	//		Location = Coord(floatCoordArray[i].x, floatCoordArray[i].y, 0);
	//		vertices.push_back(Location);
	//	}
	//	pointSet.setPoints(vertices);

	//}




}
