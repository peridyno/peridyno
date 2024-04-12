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

#include "qcheckbox.h"
#include <algorithm>
#include <QPainterPath>
#include "qlayout.h"
#include "Field.h"

#include "Curve.h"
#include "QComponent.h"

namespace dyno
{


	class QCurveWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QCurveWidget(FBase* field);
		~QCurveWidget() override {};



	public slots:
		void updateField();

	};

	class QCurveLabel : public mDrawLabel
	{
		Q_OBJECT
	
	public slots:
		void changeValue(int s);
		void changeInterpValue(int s);
		void changeLabelSize();
		void setCurveClose();
		void updateField() {};
		void setSpacingToDrawLabel(double value, int id);
		void SetValueToDrawLabel(double value, int id);
		void setLinearResample(int s);

	public:

		QCurveLabel(QWidget* parent = nullptr);
		QCurveLabel(FVar<Curve>* f, QWidget* parent = nullptr) 
		{
			this->setField(f);
			this->copySettingFromField();
			if (isSquard) 
				this->setLabelSize(w, w, w, w);
			else
				this->setLabelSize(w, h, w, w);
			this->setStyleSheet("background:rgba(110,115,100,1)");
			this->setMouseTracking(true);
		};
		~QCurveLabel();
		void updateDataToField();
		Dir setMode(int s) { return Dir(s); }
		//void setBorderMode(int border) { if (border == 0) { borderMode = Open; } else { borderMode = Close; } }
		void setField(FVar<Curve>* f)
		{
			field = TypeInfo::cast<FVar<Curve>>(f);
		}
		void copyFromField(std::vector<Curve::Coord2D> coord01, std::vector<MyCoord>& thisArray);
		void copyFromField(std::vector<Curve::OriginalCoord> coord01, std::vector<MyCoord>& thisArray);
		void updateLabelShape();
		void copySettingFromField();

	protected:
		void paintEvent(QPaintEvent* event);
		void mouseMoveEvent(QMouseEvent* event) override;
		void mousePressEvent(QMouseEvent* event)override;
		void mouseReleaseEvent(QMouseEvent* event)override;
		void reSort(std::vector<MyCoord>& vector1);
		void initializeLine(Dir mode);
		void deletePoint();
		void setLabelSize(int minX, int minY, int maxX, int maxY);
		void buildBezierPoint();
		void keyPressEvent(QKeyEvent* event);
		int addPointtoEnd();
		int insertCurvePoint(MyCoord pCoord);
		void insertHandlePoint(int fp, MyCoord pCoord);
		void keyReleaseEvent(QKeyEvent* event);
		void leaveEvent(QEvent* event);
		void remapArrayToHeight(std::vector<MyCoord>& Array, int h);
		void updateFloatCoordArray(std::vector<MyCoord> CoordArray, std::vector<Coord0_1>& myfloatCoord);
		void CoordtoField(Curve& s);
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
		bool isSquard = true;
		bool useSort = true;
		bool lockSize = false;
		bool curveClose = false;
		bool LineResample = false;

		double spacing = 10;				//Resampling Spacing

	private:

		//Default Setting
		int w = 380;						//Default Width
		int h = 100;						//Default Height

		int selectDistance = 10;			//SelectDistanceThreshold
		int HandleAngleThreshold = 20;
		int radius = 4;						//Point Radius
		float iniHandleLength = 15;			//DefaultHandleLength

		Dir Mode = x;						//Direction for CloseMode
		Interpolation InterpMode = Linear;   //Interpolation
		FVar<Curve>* field;					//field

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
