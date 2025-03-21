/**
 * Program:   Qt Property Item
 * Module:    QRampWidget.h
 *
 * Copyright 2023 Yuzhong Guo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "QFieldWidget.h"
#include "QtGUI/PPropertyWidget.h"

#include <QKeyEvent>
#include <QEvent>

#include "Field/Ramp.h"
#include "PCustomWidgets.h"
#include "QBoolFieldWidget.h"
#include "qcheckbox.h"
#include <algorithm>
#include <QPainterPath>
#include "qlayout.h"
#include "Field.h"

namespace dyno
{

	class mQDoubleSlider : public QDoubleSlider
	{
		Q_OBJECT

	public:

		mQDoubleSlider(QWidget* pParent = NULL) {};

		int id = -1;

	public:
		QLabel* nameLabel = nullptr;

	public slots:

		void setNewVisable(int s)
		{
			this->setVisible(bool(s));
			if (this->nameLabel != nullptr) 
			{
				this->nameLabel->setVisible(bool(s));
			}
		}
	};

	class mQDoubleSpinner : public QDoubleSpinner
	{
		Q_OBJECT

	public:

		mQDoubleSpinner(QWidget* pParent = NULL) 
		{
			connect(this, SIGNAL(valueChanged(double)), this, SLOT(setValue(double)));
		}

	signals:
		void valueChangedAndID(double Value, int i);

	public slots:

		void setValue(double Value, bool BlockSignals = false) 
		{

			if (!BlockSignals) { emit valueChangedAndID(Value, id); }
			QDoubleSpinner::setValue(Value,BlockSignals);

		}

		void setNewVisable(int s)
		{
			this->setVisible(bool(s));
		}


	public:
		int id = -1;
	};

	class mQCheckBox : public QCheckBox
	{
		Q_OBJECT

	public:

		mQCheckBox(QWidget* pParent = NULL)
		{
			connect(this, SIGNAL(stateChanged(int)), this, SLOT(findSignal(int)));
			
		}

	signals:

		void mValueChanged(int);

	public slots:

		void findSignal(int i)
		{
			 emit mValueChanged(i); 

		}

		void setNewVisable(int s)
		{
			this->setVisible(!bool(s));
			if (this->nameLabel != nullptr)
			{
				this->nameLabel->setVisible(!bool(s));
			}
		}

		void updateChecked(int s) 
		{
			if (s ==1 && this->checkState()== Qt::CheckState::Unchecked)
			{
				setCheckState(Qt::CheckState::Checked);
				emit stateChanged(1);
			}
		}
		void updateUnchecked(int s)
		{
			if (s == 0 && this->checkState() == Qt::CheckState::Checked)
			{
				setCheckState(Qt::CheckState::Unchecked);
				emit stateChanged(0);
			}
		}


	public:
		QLabel* nameLabel = nullptr;
	};

	//******************************* DrawLabel ********************************//
	class mDrawLabel : public QWidget
	{
		Q_OBJECT


	public:
		struct MyCoord
		{
			int x = 0;
			int y = 0;
			void set(int a, int b)
			{
				this->x = a;
				this->y = b;
			}
			void set(MyCoord s)
			{
				this->x = s.x;
				this->y = s.y;
			}
			MyCoord(int a, int b) {
				this->x = a;
				this->y = b;
			}

			MyCoord(Vec2f s) {
				this->x = s[0];
				this->y = s[1];
			}
			MyCoord() {};

			bool operator == (MyCoord s)
			{
				if (this->x == s.x && this->y == s.y)
				{
					return true;
				}
				return false;
			}
			MyCoord operator - (MyCoord s)
			{
				MyCoord a;
				a.x = this->x -= s.x;
				a.y = this->y -= s.y;
				return a;
			}
			MyCoord operator + (MyCoord s)
			{
				MyCoord a;
				a.x = this->x += s.x;
				a.y = this->y += s.y;
				return a;
			}

		};
		struct Coord0_1
		{
			double x = 0;
			double y = 0;
			void set(double a, double b)
			{
				this->x = a;
				this->y = b;
			}
		};
		struct EndPoint
		{
			int firstPoint = -1;
			int secondPoint = -1;
			EndPoint() {};
			EndPoint(int f, int s)
			{
				firstPoint = f;
				secondPoint = s;
			}
		};

		enum Dir
		{
			x = 0,
			y = 1,
		};
		enum Interpolation
		{
			Linear = 0,
			Bezier = 1,
		};

		enum BorderMode
		{
			Open = 0,
			Close = 1,
		};


	public:

		mDrawLabel() {};

		~mDrawLabel() 
		{
			//CurveCoord
			mCoordArray.clear();
			mReSortCoordArray.clear();
			mHandlePoints.clear();
			mFloatCoord.clear();
			mHandleFloatCoord.clear();
			mCurvePoint.clear();
			mMultiSelectID.clear();

			//Map
			mCurvePointMapLength.clear();
			mLengthMapEndPoint.clear();
			mMapOriginalIDtoResortID.clear();
			mMapResortIDtoOriginalID.clear();     

		};

		virtual void updateDataToField() {};

		void updateLabelShape(bool squard)
		{
			if (squard)
			{
				this->setLabelSize(w, w, w, w);
				remapArrayToHeight(mCoordArray, w);
				remapArrayToHeight(mReSortCoordArray, w);
				remapArrayToHeight(mHandlePoints, w);
			}
			else
			{
				this->setLabelSize(w, h, w, w);
				remapArrayToHeight(mCoordArray, h);
				remapArrayToHeight(mReSortCoordArray, h);
				remapArrayToHeight(mHandlePoints, h);
			}

			//ForceUpdate = true;
			this->update();
		}


	public slots:

		void setCurveClose()
		{
			curveClose = !curveClose;
			this->mForceUpdate = true;
			update();
		}


		void setLinearResample(int s)
		{
			LineResample = s;
			this->mForceUpdate = true;
			update();
		}

		void setSpacingToDrawLabel(double value, int id)
		{
			spacing = value;
			mForceUpdate = true;
			update();
		}


	protected:

		
		void copyFromField(std::vector<Canvas::Coord2D> coord01, std::vector<MyCoord>& thisArray)
		{
			if (coord01.size())
			{
				for (auto it : coord01)
				{
					Coord0_1 s;
					s.set(it.x, it.y);
					thisArray.push_back(ZeroOneToCoord(s, minX, maxX, minY, maxY));
				}
			}
		}
		void copyFromField(std::vector<Canvas::OriginalCoord> coord01, std::vector<MyCoord>& thisArray)
		{
			if (coord01.size())
			{
				for (auto it : coord01)
				{
					MyCoord s;
					s.set(it.x, it.y);
					thisArray.push_back(s);
				}
			}
		}

		void initializeLine(Dir mode)
		{
			if (mode == x)
			{
				mCoordArray[0].x = minX;
				mCoordArray[0].y = (maxY + minY) / 2;
				mCoordArray[1].x = maxX;
				mCoordArray[1].y = (maxY + minY) / 2;
			}
			if (mode == y)
			{
				mCoordArray[0].x = (maxX + minX) / 2;
				mCoordArray[0].y = minY;
				mCoordArray[1].x = (maxX + minX) / 2;
				mCoordArray[1].y = maxY;
			}
		}

		void setLabelSize(int minX, int minY, int maxX, int maxY)
		{
			this->setMinimumSize(minX, minY);
			this->setMaximumSize(maxX, maxY);
		}

		void reSort(std::vector<MyCoord>& vector1)
		{
			if (useSort)
			{
				if (mMode == x)
				{
					std::sort(vector1.begin(), vector1.end(), sortx);
				}

				if (mMode == y)
				{
					sort(vector1.begin(), vector1.end(), sorty);
				}
			}
		}

		MyCoord ZeroOneToCoord(Coord0_1& value, int x0, int x1, int y0, int y1)
		{
			MyCoord s;
			s.x = int(value.x * float(x1 - x0)) + x0;
			s.y = int((1 - value.y) * float(y1 - y0)) + y0;

			return s;
		}

		void deletePoint()
		{
			if (mMultiSelectID.size() <= 1)
			{
				mCoordArray.erase(mCoordArray.begin() + mSelectPointID);
				mHandlePoints.erase(mHandlePoints.begin() + mSelectPointID * 2 + 1);
				mHandlePoints.erase(mHandlePoints.begin() + mSelectPointID * 2);
				mReSortCoordArray.clear();
				mReSortCoordArray.assign(mCoordArray.begin(), mCoordArray.end());
				reSort(mReSortCoordArray);
				buildCoordToResortMap();
			}
			else
			{
				std::sort(mMultiSelectID.begin(), mMultiSelectID.end());

				for (size_t i = 0; i < mMultiSelectID.size(); i++)
				{
					mSelectPointID = mMultiSelectID[mMultiSelectID.size() - i - 1];
					mCoordArray.erase(mCoordArray.begin() + mSelectPointID);
					mHandlePoints.erase(mHandlePoints.begin() + mSelectPointID * 2 + 1);
					mHandlePoints.erase(mHandlePoints.begin() + mSelectPointID * 2);
					mReSortCoordArray.clear();
					mReSortCoordArray.assign(mCoordArray.begin(), mCoordArray.end());
					reSort(mReSortCoordArray);
					buildCoordToResortMap();
				}

			}

			mMultiSelectID.clear();

			mSelectPointID = -1;
			mHoverPoint = -1;
			mIsHover = false;
			mIsSelect = false;
			mHandleParent = -1;
			mSelectHandlePoint = -1;
			mConnectHandlePoint = -1;
			mIsHandleSelect = false;

		}

		virtual int addPointtoEnd() 
		{
			if (!mInsertAtBegin)
			{
				mCoordArray.push_back(mPressCoord);
				buildCoordToResortMap();
				insertHandlePoint(mCoordArray.size() - 1, mPressCoord);

				if (InterpMode == Interpolation::Bezier)
				{
					mInsertBezierOpenPoint = true;
					mSelectPointID = -1;
					mIsSelect = false;
					mHandleParent = mCoordArray.size() - 1;
					mSelectHandlePoint = mHandleParent * 2 + 1;
					mConnectHandlePoint = mSelectHandlePoint - 1;
					mIsHandleSelect = true;
				}
				if (InterpMode == Interpolation::Linear)
				{
					mInsertBezierOpenPoint = false;
					mSelectPointID = mCoordArray.size() - 1;
					mIsSelect = true;
					mHandleParent = mSelectPointID;
					mSelectHandlePoint = -1;
					mConnectHandlePoint = -1;
					mIsHandleSelect = false;
					mInitPosition.set(mCoordArray[mSelectPointID]);
				}

				mMultiSelectID.clear();
				mMultiSelectID.push_back(mCoordArray.size() - 1);
			}
			else if (mInsertAtBegin)
			{
				mCoordArray.insert(mCoordArray.begin(), mPressCoord);
				buildCoordToResortMap();
				insertHandlePoint(0, mPressCoord);


				if (InterpMode == Interpolation::Bezier)
				{
					mInsertBezierOpenPoint = true;
					mSelectPointID = -1;
					mIsSelect = false;
					mHandleParent = 0;
					mSelectHandlePoint = 1;
					mConnectHandlePoint = mSelectHandlePoint - 1;
					mIsHandleSelect = true;
				}
				if (InterpMode == Interpolation::Linear)
				{
					mInsertBezierOpenPoint = false;
					mSelectPointID = 0;
					mIsSelect = true;
					mHandleParent = mSelectPointID;
					mSelectHandlePoint = -1;
					mConnectHandlePoint = -1;
					mIsHandleSelect = false;
					mInitPosition.set(mCoordArray[mSelectPointID]);
				}
			}
			return mHandleParent;
		}

		void insertHandlePoint(int fp, MyCoord pCoord)
		{
			dyno::Vec2f P(pCoord.x, pCoord.y);
			int size = mReSortCoordArray.size();
			dyno::Vec2f p1;
			dyno::Vec2f p2;

			dyno::Vec2f N = Vec2f(1, 0);
			if (fp == 0)
			{
				if (mReSortCoordArray.size() == 1)
					N = Vec2f(1, 0);
				else
					N = Vec2f(pCoord.x - mReSortCoordArray[1].x, pCoord.y - mReSortCoordArray[1].y);
			}
			else
			{
				N = Vec2f(pCoord.x - mReSortCoordArray[fp - 1].x, pCoord.y - mReSortCoordArray[fp - 1].y) * -1;
			}

			N.normalize();

			int num = fp * 2;

			p1 = P - N * mIniHandleLength;
			p2 = P + N * mIniHandleLength;

			mHandlePoints.insert(mHandlePoints.begin() + num, MyCoord(p1));
			mHandlePoints.insert(mHandlePoints.begin() + num, MyCoord(p2));
		}

		void buildBezierPoint()
		{
			int totalLength = mPath.length();
			mCurvePoint.clear();
			mCurvePointMapLength.clear();
			for (size_t i = 0; i < 500; i++)
			{
				float length = i * spacing;
				qreal perc = 0;
				QPointF QP;
				bool b = false;
				if (length <= totalLength)
				{
					perc = mPath.percentAtLength(qreal(length));
				}
				else
				{
					perc = 1;
					b = true;
				}

				QP = mPath.pointAtPercent(perc);
				mCurvePoint.push_back(MyCoord(QP.x(), QP.y()));
				mCurvePointMapLength[i] = length;

				if (b) { break; }
			}

		}

		int insertCurvePoint(MyCoord pCoord)
		{
			mLengthMapEndPoint.clear();
			QPainterPath tempPath;
			for (size_t i = 1; i < mReSortCoordArray.size(); i++)
			{

				int ptnum = i - 1;
				tempPath.moveTo(mReSortCoordArray[ptnum].x, mReSortCoordArray[ptnum].y);

				auto it = mMapResortIDtoOriginalID.find(i);
				int id = it->second;
				int s = id * 2;

				auto itf = mMapResortIDtoOriginalID.find(ptnum);
				int idf = itf->second;
				int f = idf * 2 + 1;
				if (InterpMode == Interpolation::Bezier)
				{
					tempPath.cubicTo(QPointF(mHandlePoints[f].x, mHandlePoints[f].y), QPointF(mHandlePoints[s].x, mHandlePoints[s].y), QPointF(mReSortCoordArray[ptnum + 1].x, mReSortCoordArray[ptnum + 1].y));
				}
				else if (InterpMode == Interpolation::Linear)
				{
					tempPath.lineTo(QPointF(mReSortCoordArray[ptnum + 1].x, mReSortCoordArray[ptnum + 1].y));
				}
				float tempLength = tempPath.length();

				EndPoint tempEP = EndPoint(id, idf);
				mLengthMapEndPoint[tempLength] = tempEP;
			}

			int dis = 380;
			int nearPoint = -1;
			int temp;

			for (size_t i = 0; i < mCurvePoint.size(); i++)
			{
				temp = sqrt(std::pow((pCoord.x - mCurvePoint[i].x), 2) + std::pow((pCoord.y - mCurvePoint[i].y), 2));

				if (dis >= temp)
				{
					nearPoint = i;
					dis = temp;
				}
			}

			int fp = -1;
			int searchRadius = 20; 

			if (dis < searchRadius)
			{
				float realLength = mCurvePointMapLength.find(nearPoint)->second;
				float finalLength = -1;
				for (auto it : mLengthMapEndPoint)
				{
					if (realLength <= it.first)
					{
						finalLength = it.first;
						break;
					}
				}
				if (finalLength == -1)
				{
					fp = addPointtoEnd();
				}
				else
				{
					fp = mLengthMapEndPoint.find(finalLength)->second.firstPoint;

					mCoordArray.insert(mCoordArray.begin() + fp, pCoord);
					mReSortCoordArray.clear();
					mReSortCoordArray.assign(mCoordArray.begin(), mCoordArray.end());
					reSort(mReSortCoordArray);
					buildCoordToResortMap();
					insertHandlePoint(fp, pCoord);

					mInitPosition.set(pCoord);
				}
			}
			else
			{
				fp = addPointtoEnd();
			}
			return fp;

		}


		static bool sortx(MyCoord a, MyCoord b)
		{
			if (a.x < b.x) { return true; }
			else { return false; }
		};
		static bool sorty(MyCoord a, MyCoord b)
		{
			if (a.y < b.y) { return true; }
			else { return false; }
		};

		void buildCoordToResortMap()
		{
			mReSortCoordArray.assign(mCoordArray.begin(), mCoordArray.end());
			reSort(mReSortCoordArray);

			for (size_t i = 0; i < mReSortCoordArray.size(); i++)
			{
				for (size_t k = 0; k < mCoordArray.size(); k++)
				{
					if (mReSortCoordArray[i] == mCoordArray[k])
					{
						mMapOriginalIDtoResortID[k] = i;
						mMapResortIDtoOriginalID[i] = k;
						break;
					}
				}
			}

		};

		void  keyReleaseEvent(QKeyEvent* event)
		{
			QWidget::keyPressEvent(event);
			parent()->event((QEvent*)event);
			if (event->key() == Qt::Key_Alt)
			{
				mAltKey = false;
				return;
			}
			if (event->key() == Qt::Key_Control)
			{
				mCtrlKey = false;
				return;
			}
			if (event->key() == Qt::Key_Shift)
			{
				mShiftKey = false;
				return;
			}
		}



		void insertElementToHandlePointSet(int i)
		{
			dyno::Vec2f P(mCoordArray[i].x, mCoordArray[i].y);

			dyno::Vec2f p1;
			dyno::Vec2f p2;
			dyno::Vec2f N;

			auto it = mMapOriginalIDtoResortID.find(i);
			int id = it->second;
			int f;
			int s;
			if (mCoordArray.size() < 3)
			{
				N = Vec2f(1, 0);
			}
			else
			{
				f = id - 1;
				s = id + 1;
				if (id == 0)
				{
					N[0] = mReSortCoordArray[s].x - mReSortCoordArray[id].x;
					N[1] = mReSortCoordArray[s].y - mReSortCoordArray[id].y;
				}
				else if (id == mReSortCoordArray.size() - 1)
				{
					N[0] = mReSortCoordArray[id].x - mReSortCoordArray[f].x;
					N[1] = mReSortCoordArray[id].y - mReSortCoordArray[f].y;
				}
				else
				{
					N[0] = mReSortCoordArray[s].x - mReSortCoordArray[f].x;
					N[1] = mReSortCoordArray[s].y - mReSortCoordArray[f].y;
				}
			}

			N.normalize();

			p1 = P - N * mIniHandleLength;
			p2 = P + N * mIniHandleLength;

			mHandlePoints.push_back(MyCoord(p1));
			mHandlePoints.push_back(MyCoord(p2));

		}

		void buildHandlePointSet()
		{
			for (size_t i = 0; i < mCoordArray.size(); i++)
			{
				insertElementToHandlePointSet(i);
			}
		}

		Coord0_1 CoordTo0_1Value(MyCoord& coord)
		{
			//曲线坐标转换到0-1浮点值，并反转Y轴
			Coord0_1 s;

			double x = double(coord.x);
			double y = double(coord.y);
			double fmaxX = double(maxX);
			double fminX = double(minX);
			double fmaxY = double(maxY);
			double fminY = double(minY);

			s.x = (x - fminX) / (fmaxX - fminX);
			s.y = 1 - (y - fminY) / (fmaxY - fminY);

			return s;
		}

		void leaveEvent(QEvent* event)
		{
			mShiftKey = false;
			mAltKey = false;
			this->releaseKeyboard();
		}

		void remapArrayToHeight(std::vector<MyCoord>& Array, int h)
		{
			double fmaxX = double(maxX);
			double fminX = double(minX);
			double fmaxY = double(maxY);
			double fminY = double(minY);
			for (size_t i = 0; i < Array.size(); i++)
			{
				int newMaxY = h - 1.5 * double(radius);
				float k = (double(Array[i].y) - fminY) / (fmaxY - fminY);
				Array[i].y = k * (newMaxY - fminY) + fminY;
			}
		}

		void updateFloatCoordArray(std::vector<MyCoord> CoordArray, std::vector<Coord0_1>& myfloatCoord)
		{
			myfloatCoord.clear();
			for (auto it : CoordArray)
			{
				myfloatCoord.push_back(CoordTo0_1Value(it));
			}
		}

		void keyPressEvent(QKeyEvent* event)
		{
			QWidget::keyPressEvent(event);
			parent()->event((QEvent*)event);
			if (event->key() == Qt::Key_Alt)
			{
				mAltKey = true;
				return;
			}
			if (event->key() == Qt::Key_Control)
			{

				mCtrlKey = true;
				return;
			}
			if (event->key() == Qt::Key_Shift)
			{
				mShiftKey = true;
				return;
			}
		}

		void mouseReleaseEvent(QMouseEvent* event)
		{
			mSelectPointID = -1;
			mIsSelect = false;
			mIsHandleSelect = false;
			mInsertBezierOpenPoint = false;
			mSelectHandlePoint = -1;
			mConnectHandlePoint = -1;
			mHandleConnection = true;

			//按照百分比划Bezier分点
			buildBezierPoint();
			updateDataToField();
			update();
		}


	public:
		//Setting
		bool useBezier = false;

		bool useSort = true;
		bool lockSize = false;
		bool curveClose = false;
		bool LineResample = false;

		double spacing = 10;

	protected:

		//Default Setting
		int w = 380;						//Default Width
		int h = 100;						//Default Height

		int selectDistance = 10;			//SelectDistanceThreshold
		int HandleAngleThreshold = 20;
		int radius = 4;						//Point Radius
		float mIniHandleLength = 15;			//DefaultHandleLength

		Dir mMode = x;						//Direction for CloseMode
		Interpolation InterpMode = Linear;   //Interpolation

		//reSize
		int maxX = 0;
		int maxY = 0;
		int minX = 0;
		int minY = 0;

		bool mGeneratorXmin = 1;
		bool mGeneratorXmax = 1;
		bool mGeneratorYmin = 1;
		bool mGeneratorYmax = 1;

		//Temp
		int mMinIndex = 0;
		int mMaxIndex = 0;

		//KeyboardStatus
		bool mAltKey = false;
		bool mCtrlKey = false;
		bool mShiftKey = false;

		//CurrentStatus
		int mHoverPoint = -1;
		int mSelectHandlePoint = -1;
		int mSelectPointID = -1;
		bool mIsSelect = false;
		bool mIsHandleSelect = false;
		int mHandleParent = -1;
		int mSelectPointInResort = -1;
		bool mIsHover = false;
		int mConnectHandlePoint = -1;
		float mConnectLength;
		bool mForceUpdate = false;
		bool mInsertBezierOpenPoint = false;
		bool mHandleConnection = true;
		bool mInsertAtBegin = false;

		//Coord
		MyCoord mInitPosition;					//initial position of the SelectedPoint
		MyCoord mDtPosition;			//delta position of th SelectedPoint
		MyCoord mInitHandlePos;			//initial position of the SelectedHandlePoint
		MyCoord mPressCoord;				//Current PressCoord


		//CurveCoord
		std::vector<MyCoord> mCoordArray;			//CoordArray by Press
		std::vector<MyCoord> mReSortCoordArray;		//Reorder Coordarray
		std::vector<MyCoord> mHandlePoints;			//handlePoints by CoordArray
		std::vector<Coord0_1> mFloatCoord;			// updateFloatCoordArray(reSortCoordArray, floatCoord); 
		std::vector<Coord0_1> mHandleFloatCoord;
		std::vector<MyCoord> mCurvePoint;			//	resample Path
		std::vector<int> mMultiSelectID;

		//Map
		std::map<int, float> mCurvePointMapLength;		//find Length by curvePoint
		std::map<float, EndPoint> mLengthMapEndPoint;	//find EndPoint by Length
		std::map<int, int> mMapOriginalIDtoResortID;		//find reSortArray id by CoordArray id
		std::map<int, int> mMapResortIDtoOriginalID;     //find CoordArray id by reSortArray id

		//Path
		QPainterPath mPath;
	};


}
