#include "Vector/Vector3D.h"
#include "Vector/Vector2D.h"

#include "Curve.h"

namespace dyno {

	Curve::Curve()
	{

	}


	Curve::Curve(const Curve& curve)
	{

		this->mCoord = curve.mCoord;

		this->mFinalCoord = curve.mFinalCoord;
		this->Originalcoord = curve.Originalcoord;
		this->OriginalHandlePoint = curve.OriginalHandlePoint;

		this->mBezierPoint = curve.mBezierPoint;
		this->mResamplePoint = curve.mResamplePoint;
		this->myHandlePoint = curve.myHandlePoint;

		this->mLengthArray = curve.mLengthArray;
		this->length_EndPoint_Map = curve.length_EndPoint_Map;

		this->mInterpMode = curve.mInterpMode;

		this->remapRange[8] = curve.mInterpMode;// "MinX","MinY","MaxX","MaxY"

		this->lockSize = curve.lockSize;
		this->useCurve = curve.useCurve;

		this->useSquard = curve.useSquard;
		this->curveClose = curve.curveClose;
		this->resample = curve.resample;

		this->useColseButton = curve.useColseButton;
		this->useSquardButton = curve.useSquardButton;

		this->Spacing = curve.Spacing;

		this->NminX = curve.NminX;
		this->NmaxX = curve.NmaxX;
		this->NminY = curve.NminY;
		this->NmaxY = curve.NmaxY;

		this->segment = curve.segment;
		this->resampleResolution = curve.resampleResolution;

	}

	// Update FinalCoord
	void Curve::UpdateFieldFinalCoord()
	{

		mFinalCoord.clear();

		//如果使用贝塞尔插值
		if (mInterpMode == Interpolation::Bezier )
		{
			if (mCoord.size() >= 2)
			{
				updateBezierCurve();
			}
			if (resample)
			{
				std::vector<Coord2D> myBezierPoint_H;
				updateResampleBezierCurve(myBezierPoint_H);
				resamplePointFromLine(myBezierPoint_H);
				
				mFinalCoord.assign(mResamplePoint.begin(), mResamplePoint.end());
			}

		}
		//如果使用线性插值
		else if (mInterpMode == Interpolation::Linear )
		{
			if (resample)
			{
				if (mCoord.size() >= 2)
				{

					updateResampleLinearLine();
				}
				mFinalCoord.assign(mResamplePoint.begin(), mResamplePoint.end());
			}
			else
			{
				mFinalCoord.assign(mCoord.begin(), mCoord.end());
			}
		}


		for (size_t i = 0; i < mFinalCoord.size(); i++)
		{
			mFinalCoord[i].x = (NmaxX - NminX) * mFinalCoord[i].x + NminX;
			mFinalCoord[i].y = (NmaxY - NminY) * mFinalCoord[i].y + NminY;
		}


	}

	//C++ Edit





	// C++ Bezier 插值Bezier曲线



	


	void Curve::updateResampleBezierCurve(std::vector<Coord2D>& myBezierPoint_H)
	{
		float temp = segment;
		segment = resampleResolution;
		myBezierPoint_H.clear();

		int n = mCoord.size();
		int bn = myHandlePoint.size();

		//如果手柄数目不对，重新构建贝塞尔手柄
		if (bn != 2 * n)
		{
			rebuildHandlePoint(mCoord);
		}

		//遍历每个线段，按照“F点、F点手柄2、E点手柄1、E点”  插值贝塞尔曲线点
		for (int i = 0; i < n - 1; i++)
		{
			Coord2D p0 = mCoord[i];
			Coord2D p1 = myHandlePoint[2 * i + 1];
			Coord2D p2 = myHandlePoint[2 * (i + 1)];
			Coord2D p3 = mCoord[i + 1];
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint_H);
		}
		//如果曲线闭合，且能形成完整环形。则将第一个点、最后一个点按照上述逻辑继续   插值贝塞尔曲线点
		if (curveClose && n >= 3)
		{
			Coord2D p0 = mCoord[n - 1];
			Coord2D p1 = myHandlePoint[bn - 1];
			Coord2D p2 = myHandlePoint[0];
			Coord2D p3 = mCoord[0];
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint_H);
		}
		//如果曲线闭合
		if (curveClose)
		{
			if (mCoord.size())
			{
				myBezierPoint_H.push_back(mCoord[0]);
			}
		}
		//如果曲线不闭合，将最后一个点返回
		else
		{
			if (mCoord.size())
			{
				myBezierPoint_H.push_back(mCoord[mCoord.size() - 1]);
			}
		}

		buildSegMent_Length_Map(myBezierPoint_H);

		segment = temp;
	}




	//bezier point





	//widget to field;
	//void Curve::addItemMyCoord(float x, float y)
	//{
	//	Coord2D s;
	//	s.set(x, y);
	//	MyCoord.push_back(s);
	//}












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
