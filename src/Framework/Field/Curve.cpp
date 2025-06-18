#include "Vector/Vector3D.h"
#include "Vector/Vector2D.h"

#include "Curve.h"

namespace dyno {

	Curve::Curve()
	{

	}


	Curve::Curve(const Curve& curve)
	{

		this->mUserCoord = curve.mUserCoord;

		this->mFinalCoord = curve.mFinalCoord;

		this->mBezierPoint = curve.mBezierPoint;
		this->mResamplePoint = curve.mResamplePoint;
		this->mUserHandle = curve.mUserHandle;

		this->mLengthArray = curve.mLengthArray;
		this->mLength_EndPoint_Map = curve.mLength_EndPoint_Map;

		this->mInterpMode = curve.mInterpMode;


		this->mClose = curve.mClose;
		this->mResample = curve.mResample;

		this->mSpacing = curve.mSpacing;


	}

	// Update FinalCoord
	void Curve::UpdateFieldFinalCoord()
	{

		mFinalCoord.clear();

		//Bezier Mode
		if (mInterpMode == Interpolation::Bezier )
		{
			if (mUserCoord.size() >= 2)
			{
				updateBezierCurve();
			}
			if (mResample)
			{
				std::vector<Coord2D> myBezierPoint_H;
				updateResampleBezierCurve(myBezierPoint_H);
				resamplePointFromLine(myBezierPoint_H);
				
				mFinalCoord.assign(mResamplePoint.begin(), mResamplePoint.end());
			}

		}

		//LinearMode
		else if (mInterpMode == Interpolation::Linear )
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



	void Curve::updateResampleBezierCurve(std::vector<Coord2D>& myBezierPoint_H)
	{

		myBezierPoint_H.clear();

		int n = mUserCoord.size();
		int bn = mUserHandle.size();

		//check handle number
		if (bn != 2 * n)
		{
			rebuildHandlePoint(mUserCoord);
		}

		//build bezier
		for (int i = 0; i < n - 1; i++)
		{
			Coord2D p0 = mUserCoord[i];
			Coord2D p1 = mUserHandle[2 * i + 1];
			Coord2D p2 = mUserHandle[2 * (i + 1)];
			Coord2D p3 = mUserCoord[i + 1];
			updateBezierPointToBezierSet(p0, p1, p2, p3, myBezierPoint_H);
		}
		//close bezier
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

}
