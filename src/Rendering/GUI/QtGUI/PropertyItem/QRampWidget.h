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

#include "Ramp.h"
#include "PCustomWidgets.h"
#include "QBoolFieldWidget.h"
#include "qcheckbox.h"
#include <algorithm>
#include <QPainterPath>
#include "qlayout.h"
#include "Field.h"

namespace dyno
{


	class QRampWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QRampWidget(FBase* field);
		~QRampWidget() override {};
	


	public slots:
		void updateField();

	};

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


	public:
		QLabel* nameLabel = nullptr;
	};

	class QDrawLabel : public QWidget
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
			MyCoord (int a, int b) {
				this->x = a;
				this->y = b;
			}

			MyCoord (Vec2f s) {
				this->x = s[0];
				this->y = s[1];
			}
			MyCoord() {};

			bool operator == (MyCoord s ) 
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

	public slots:
		void changeValue(int s);
		void changeInterpValue(int s);
		void changeLabelSize();
		void setCurveClose();
		void updateField() {};
		void setSpacingToDrawLabel(double value, int id);
		void SetValueToDrawLabel(double value, int id);
		void setLinearResample(int s);
		void setUseRamp(int v);

	public:

		QDrawLabel(QWidget* parent = nullptr);
		QDrawLabel(FVar<Ramp>* f,QWidget* parent = nullptr);
		~QDrawLabel();
		void updateDataToField();
		Dir setMode(int s) { return Dir(s); }
		void setBorderMode(int border) { if (border == 0) {borderMode = Open;}else{borderMode = Close;}}
		void setField(FVar<Ramp>* f) 
		{
			field = TypeInfo::cast<FVar<Ramp>>(f);
		}
		void copyFromField(std::vector<Ramp::Coord2D> coord01, std::vector<MyCoord>& thisArray);
		void copyFromField(std::vector<Ramp::OriginalCoord> coord01, std::vector<MyCoord>& thisArray);
		void updateLabelShape();
		void copySettingFromField();

	protected:
		void paintEvent(QPaintEvent* event);
		void mouseMoveEvent(QMouseEvent* event) override;
		void mousePressEvent(QMouseEvent* event)override;
		void mouseReleaseEvent(QMouseEvent* event)override;
		void reSort(std::vector<MyCoord>& vector1);
		void initializeLine(Dir mode );
		void deletePoint();
		void setLabelSize(int minX , int minY, int maxX, int maxY);
		void buildBezierPoint();
		void keyPressEvent(QKeyEvent* event);
		int addPointtoEnd();
		int insertCurvePoint(MyCoord pCoord);
		void insertHandlePoint(int fp, MyCoord pCoord);	
		void keyReleaseEvent(QKeyEvent* event);
		void leaveEvent(QEvent* event);
		void remapArrayToHeight(std::vector<MyCoord>& Array,int h);
		void updateFloatCoordArray(std::vector<MyCoord> CoordArray, std::vector<Coord0_1>& myfloatCoord);
		void CoordtoField(Ramp& s);
		Coord0_1 CoordTo0_1Value(MyCoord& coord);
		MyCoord ZeroOneToCoord(Coord0_1& value, int x0, int x1, int y0, int y1);
		void buildCoordToResortMap();
		void buildHandlePointSet();
		void insertElementToHandlePointSet(int i);

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



	public:
		//Setting
		bool useBezier = false;
		bool isSquard = false;
		bool useSort = true;
		bool lockSize = false;
		bool curveClose = false;
		bool LineResample = false;
		bool useRamp = false;
		double spacing = 10;				//Resampling Spacing
		BorderMode borderMode = Close;		//Ramp Or Canvas

	private:

		//Default Setting
		int w = 380;						//Default Width
		int h = 100;						//Default Height

		int selectDistance = 10;			//SelectDistanceThreshold
		int HandleAngleThreshold = 20;  
		int radius = 4;						//Point Radius
		float iniHandleLength = 15;			//DefaultHandleLength

		Dir Mode = x;						//Direction for CloseMode
		Interpolation InterpMode= Linear;   //Interpolation
		FVar<Ramp>* field;					//field

		//reSize
		int maxX = 0;
		int maxY = 0;
		int minX = 0;
		int minY = 0;
		double NmaxX = 0;
		double NmaxY = 0;
		double NminX = 0;
		double NminY = 0;
		bool generatorXmin = 1;
		bool generatorXmax = 1;
		bool generatorYmin = 1;
		bool generatorYmax = 1;

		//Temp
		int minIndex = 0;
		int maxIndex = 0;

		//KeyboardStatus
		bool altKey = false;
		bool ctrlKey = false;
		bool shiftKey = false;

		//CurrentStatus
		int hoverPoint = -1;
		int selectHandlePoint = -1;
		int selectPoint = -1;
		bool isSelect = false;
		bool isHandleSelect = false;
		int handleParent = -1;
		int selectPointInResort = -1;
		bool isHover = false;
		int connectHandlePoint = -1;
		float connectLength;
		bool ForceUpdate = false;
		bool InsertBezierOpenPoint = false;
		bool HandleConnection = true;
		bool InsertAtBegin = false;

		//Coord
		MyCoord iniPos;					//initial position of the SelectedPoint
		MyCoord dtPos;					//delta position of th SelectedPoint
		MyCoord iniHandlePos;			//initial position of the SelectedHandlePoint
		MyCoord pressCoord;				//Current PressCoord
		MyCoord hoverCoord;				
		MyCoord selectCoord;

		//CurveCoord
		std::vector<MyCoord> CoordArray;			//CoordArray by Press
		std::vector<MyCoord> reSortCoordArray;		//Reorder Coordarray
		std::vector<MyCoord> HandlePoints;			//handlePoints by CoordArray
		std::vector<Coord0_1> floatCoord;			// updateFloatCoordArray(reSortCoordArray, floatCoord); 
		std::vector<Coord0_1> bezierFloatCoord;		// updateFloatCoordArray(CurvePoint, bezierFloatCoord); 
		std::vector<Coord0_1> handleFloatCoord;
		std::vector<MyCoord> CurvePoint;			//	resample Path
		std::vector<int> multiSelectID;				

		//Map
		std::map<int, float> curvePointMapLength;		//find Length by curvePoint
		std::map<float, EndPoint> lengthMapEndPoint;	//find EndPoint by Length
		std::map<int, int> mapOriginalIDtoResortID;		//find reSortArray id by CoordArray id
		std::map<int, int> mapResortIDtoOriginalID;     //find CoordArray id by reSortArray id

		//Path
		QPainterPath path;
	};



}
