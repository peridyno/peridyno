#include "Ramp.h"

namespace dyno {

	Ramp::Ramp() 
	{

	}

	Ramp::Ramp(Direction dir)
	{ 
		Dirmode= dir;
		useSquard = false;
	}

	Ramp::Ramp(const Ramp& ramp)
	{
		useSquard = false;
		this->Dirmode = ramp.Dirmode;
		//this->Bordermode = ramp.Bordermode;
		this->mCoord = ramp.mCoord;
		this->FE_MyCoord = ramp.FE_MyCoord;
		this->FE_HandleCoord = ramp.FE_HandleCoord;
		this->mFinalCoord = ramp.mFinalCoord;
		this->Originalcoord = ramp.Originalcoord;
		this->OriginalHandlePoint = ramp.OriginalHandlePoint;

		this->mBezierPoint = ramp.mBezierPoint;
		this->myBezierPoint_H = ramp.myBezierPoint_H;
		this->mResamplePoint = ramp.mResamplePoint;
		this->myHandlePoint = ramp.myHandlePoint;

		this->mLengthArray = ramp.mLengthArray;
		this->length_EndPoint_Map = ramp.length_EndPoint_Map;

		this->mInterpMode = ramp.mInterpMode;

		this->remapRange[8] = ramp.mInterpMode;// "MinX","MinY","MaxX","MaxY"

		this->lockSize = ramp.lockSize;
		this->useCurve = ramp.useCurve;

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

		this->segment = ramp.segment;
		this->resampleResolution = ramp.resampleResolution;

		this->xLess = ramp.xLess;
		this->xGreater = ramp.xGreater;
		this->yLess = ramp.yLess;
		this->yGreater = ramp.yGreater;

	}

	float Ramp::getCurveValueByX(float inputX)
	{
		if (mFinalCoord.size())
		{
			int l = mFinalCoord.size();
			for (size_t i = 0; i < l;i ++)
			{
				xLess = (mFinalCoord[i].x > inputX) ? xLess : mFinalCoord[i].x;
				yLess = (mFinalCoord[i].x > inputX) ? yLess : mFinalCoord[i].y;

				xGreater = (mFinalCoord[l - i - 1].x < inputX) ? xGreater : mFinalCoord[l - i - 1].x;
				yGreater = (mFinalCoord[l - i - 1].x < inputX) ? yGreater : mFinalCoord[l - i - 1].y;
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






	// C++ Bezier
	void Ramp::updateBezierCurve()
	{
		Canvas::updateBezierCurve();

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



	void Ramp::updateResampleBezierCurve() 
	{
		float temp = segment;
		segment = resampleResolution;
		myBezierPoint_H.clear();

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
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint_H);
		}
		if (curveClose && n >= 3)
		{
			Coord2D p0 = mCoord[n - 1];
			Coord2D p1 = myHandlePoint[bn - 1];
			Coord2D p2 = myHandlePoint[0];
			Coord2D p3 = mCoord[0];
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint_H);
		}
		if (curveClose)
		{
			if (mCoord.size())
			{
				myBezierPoint_H.push_back(mCoord[0]);
			}
		}
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



	//widget to field;







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

		for (int i = 0; i < mCoord.size(); i++)
		{
			if (mCoord[i].x == 0)
			{
				Cfirst = mCoord[i];
				F1 = tempHandle[2 * i];
				F2 = tempHandle[2 * i + 1];
			}
			else if (mCoord[i].x == 1)
			{
				Cend = mCoord[i];
				E1 = tempHandle[2 * i];
				E2 = tempHandle[2 * i + 1];
			}
			else
			{
				FE_MyCoord.push_back(mCoord[i]);
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

		borderCloseResort();


		mFinalCoord.clear();

		if (useCurve )
		{
			if (resample) 
			{
				if (mCoord.size() >= 2)
				{
					updateBezierCurve();
				}
				mFinalCoord.assign(mResamplePoint.begin(), mResamplePoint.end());
			}
			else 
			{
				mFinalCoord.assign(mBezierPoint.begin(), mBezierPoint.end());
			}
		}
		else if ( !useCurve ) 
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
}
