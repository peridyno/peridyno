#include "Ramp.h"

namespace dyno {

	Ramp::Ramp() 
	{

	}

	Ramp::Ramp(const Ramp& ramp)
	{

		this->mUserCoord = ramp.mUserCoord;
		this->FE_MyCoord = ramp.FE_MyCoord;
		this->FE_HandleCoord = ramp.FE_HandleCoord;
		this->mFinalCoord = ramp.mFinalCoord;

		this->mBezierPoint = ramp.mBezierPoint;
		this->myBezierPoint_H = ramp.myBezierPoint_H;
		this->mResamplePoint = ramp.mResamplePoint;
		this->mUserHandle = ramp.mUserHandle;

		this->mLengthArray = ramp.mLengthArray;
		this->mLength_EndPoint_Map = ramp.mLength_EndPoint_Map;

		this->mInterpMode = ramp.mInterpMode;

		this->mClose = ramp.mClose;
		this->mResample = ramp.mResample;

		this->mSpacing = ramp.mSpacing;

	}

	float Ramp::getCurveValueByX(float inputX)
	{
		float xLess = 1;
		float xGreater = 0;
		float yLess = 1;
		float yGreater = 0;

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

		if (mResample)
		{
			if (mInterpMode == Canvas::Interpolation::Bezier)
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

		myBezierPoint_H.clear();

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
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint_H);
		}
		if (mClose && n >= 3)
		{
			Coord2D p0 = mUserCoord[n - 1];
			Coord2D p1 = mUserHandle[bn - 1];
			Coord2D p2 = mUserHandle[0];
			Coord2D p3 = mUserCoord[0];
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint_H);
		}
		if (mClose)
		{
			if (mUserCoord.size())
			{
				myBezierPoint_H.push_back(mUserCoord[0]);
			}
		}
		else
		{
			if (mUserCoord.size())
			{
				myBezierPoint_H.push_back(mUserCoord[mUserCoord.size() - 1]);
			}
		}

		buildSegMent_Length_Map(myBezierPoint_H);

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


	void Ramp::borderCloseResort() 
	{

		std::vector<Coord2D> tempHandle;
		tempHandle.assign(mUserHandle.begin(), mUserHandle.end());
		FE_HandleCoord.clear();
		FE_MyCoord.clear();

		Coord2D Cfirst(-1, -1, -1);
		Coord2D Cend(-1, -1, -1);
		Coord2D F1(-1, -1, -1);
		Coord2D F2(-1, -1, -1);
		Coord2D E1(-1, -1, -1);
		Coord2D E2(-1, -1, -1);

		for (int i = 0; i < mUserCoord.size(); i++)
		{
			if (mUserCoord[i].x == 0)
			{
				Cfirst = mUserCoord[i];
				F1 = tempHandle[2 * i];
				F2 = tempHandle[2 * i + 1];
			}
			else if (mUserCoord[i].x == 1)
			{
				Cend = mUserCoord[i];
				E1 = tempHandle[2 * i];
				E2 = tempHandle[2 * i + 1];
			}
			else
			{
				FE_MyCoord.push_back(mUserCoord[i]);
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

		if (mInterpMode == Canvas::Interpolation::Bezier)
		{
			if (mResample)
			{
				if (mUserCoord.size() >= 2)
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
		else if (mInterpMode == Canvas::Interpolation::Linear)
		{

			if (mResample)
			{
				if (mUserCoord.size() >= 2)
				{

					updateResampleLinearLine();
				}
				mFinalCoord.assign(mResamplePoint.begin(), mResamplePoint.end());
			}
			else
			{
				mFinalCoord.assign(mUserCoord.begin(), mUserCoord.end());
			}
		}

	}
}
