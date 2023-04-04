/**
 * Program:   Qt Property Item
 * Module:    PPropertyItem.h
 *
 * Copyright 2022 Xiaowei He
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
#include <QGroupBox>

#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QMouseEvent>
#include <QDialog>
#include <QLabel>
#include "qgraphicsitem.h"
#include <QPushButton>
#include <QKeyEvent>
#include "Ramp.h"
#include <memory>

#include "FCallbackFunc.h"

namespace dyno
{
	class Node;
	class Module;
	class FBase;

	class QDoubleSpinner;
	class QDoubleSlider;

	class QBoolFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QBoolFieldWidget(FBase* field);
		~QBoolFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(int status);

	private:
		FBase* mField = nullptr;
	};

	class QIntegerFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QIntegerFieldWidget(FBase* field);
		~QIntegerFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(int);

	private:
		FBase* mField = nullptr;
	};

	class QUIntegerFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QUIntegerFieldWidget(FBase* field);
		~QUIntegerFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(int);

	private:
		FBase* mField = nullptr;
	};

	class QRealFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QRealFieldWidget(FBase* field);
		~QRealFieldWidget();



	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(double);

	private:
		void fieldUpdated();

		FBase* mField = nullptr;

		QDoubleSlider* slider = nullptr;

		std::shared_ptr<FCallBackFunc> callback = nullptr;
	};

	class ValueButton : public QPushButton
	{
		Q_OBJECT

	public:
		explicit ValueButton(QWidget* parent = nullptr);

		void mouseMoveEvent(QMouseEvent* event) override;

		void mousePressEvent(QMouseEvent* event) override;

		void mouseReleaseEvent(QMouseEvent* event) override;


		double SpinBoxData = 0;

		double Data1 = 0;
		double Data2 = 0;
		double Data3 = 0;

		double defaultValue = 0;
		double finalValue = 0;
		int StartX = 0;
		int EndX = 0;
		QDialog* parentDialog;
		QDoubleSpinBox* DSB1;
		QDoubleSpinBox* DSB2;
		QDoubleSpinBox* DSB3;

		QSpinBox* SB1;
		QSpinBox* SB2;
		QSpinBox* SB3;
		bool shiftPress = 0;

	Q_SIGNALS:
		void ValueChange(double);
	Q_SIGNALS:
		void Release(double);

	private:
		double sub = 0;
		int temp = 0;
		std::string str;
		QString text;
	};


	class ValueDialog : public QDialog 
	{
		Q_OBJECT

	public:
		explicit ValueDialog(QWidget* parent = nullptr);

		ValueDialog(double Data,QWidget* parent = nullptr);

		void mouseMoveEvent(QMouseEvent* event) override;

		void mouseReleaseEvent(QMouseEvent* event) override;

		void keyPressEvent(QKeyEvent* event) override;
		void keyReleaseEvent(QKeyEvent* event) override;

		ValueButton* button[5];

		QDoubleSpinBox* SpinBox1;
		QDoubleSpinBox* SpinBox2;
		QDoubleSpinBox* SpinBox3;

		QSpinBox* SBox1;
		QSpinBox* SBox2;
		QSpinBox* SBox3;

	Q_SIGNALS:
		void DiaValueChange(double);


	public slots:
		void ModifyValue(double);

		void initData(double);

	private:

	};

	class mDoubleSpinBox : public QDoubleSpinBox
	{
		Q_OBJECT
	public:
		explicit mDoubleSpinBox(QWidget* parent = nullptr);

		ValueDialog* ValueModify;

		bool ModifyByDialog;

		QDoubleSpinBox* DSB1;
		QDoubleSpinBox* DSB2;
		QDoubleSpinBox* DSB3;

	private:
		//Prohibited to use
		void wheelEvent(QWheelEvent* event);

		void mousePressEvent(QMouseEvent* event) override;

		void mouseReleaseEvent(QMouseEvent* event) override;

		void mouseMoveEvent(QMouseEvent* event) override;
		
		void buildDialog();

		void contextMenuEvent(QContextMenuEvent* event) override;
	signals:
	public slots:
		void ModifyValue(double);
	};

	class QVector3FieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QVector3FieldWidget(FBase* field);
		~QVector3FieldWidget();

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(double);

	private:
		void fieldUpdated();

		FBase* mField = nullptr;

		mDoubleSpinBox* spinner1;
		mDoubleSpinBox* spinner2;
		mDoubleSpinBox* spinner3;

		std::shared_ptr<FCallBackFunc> callback = nullptr;
	};

	class QVector3iFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QVector3iFieldWidget(FBase* field);
		~QVector3iFieldWidget();

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(int);

	private:
		void fieldUpdated();

		FBase* mField = nullptr;

		QSpinBox* spinner1;
		QSpinBox* spinner2;
		QSpinBox* spinner3;

		std::shared_ptr<FCallBackFunc> callback = nullptr;
	};

	class QStringFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QStringFieldWidget(FBase* field);
		~QStringFieldWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(QString str);

	private:
		FBase* mField = nullptr;

		QLineEdit* fieldname;
	};

	class QFilePathWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QFilePathWidget(FBase* field);
		~QFilePathWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue(QString str);

	private:
		FBase* mField = nullptr;

		QLineEdit* location;
	};

	class QEnumFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QEnumFieldWidget(FBase* field);
		~QEnumFieldWidget() { mComboxIndexMap.clear(); }

	public slots:
		void changeValue(int index);

	private:
		FBase* mField = nullptr;

		std::map<int, int> mComboxIndexMap;
	};

	class QStateFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QStateFieldWidget(FBase* field);
		~QStateFieldWidget() {};

	Q_SIGNALS:
		void stateUpdated(FBase* field, int status);

	public slots:
		void tagAsOuput(int status);

	private:
		FBase* mField = nullptr;
	};

	class QRampWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QRampWidget(FBase* field);
		~QRampWidget() {};

	Q_SIGNALS:
		void fieldChanged();

	public slots:
		void changeValue();

	private:
		FBase* mField = nullptr;

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
		};
		struct Coord0_1
		{
			float x = 0;
			float y = 0;
			void set(float a, float b) 
			{
				this->x = a;
				this->y = b;
			}
		};

		enum Dir
		{
			x = 0,
			y = 1,
		};

		enum BorderMode
		{
			Open = 0,
			Close = 1,
		};

		enum RectShape
		{
			Rect = 0,
			Squard= 1,

		};

	public:
		QDrawLabel(QWidget* parent = nullptr);
		~QDrawLabel();
		Dir setMode(int s) { return Dir(s); }
		void setBorderMode(int border) { if (border == 0) {borderMode = Open;}else{borderMode = Close;}}
		void setField(FVar<Ramp>* f) { field = f; }
		void copyFromField(std::vector<Ramp::MyCoord2D> coord01) 
		{
			auto ff = field->getDataPtr();
			if (coord01.size()) 
			{
				for (auto it : coord01)
				{
					Coord0_1 s;
					s.set(it.x, it.y);
					reSortCoordArray.push_back(ZeroOneToCoord(s,ff->oMinX, ff->oMaxX, ff->oMinY, ff->oMaxY));
					//printf("RCApush值：%d\n", ZeroOneToCoord(s, ff->oMinX, ff->oMaxX, ff->oMinY, ff->oMaxY).x, ZeroOneToCoord(s, ff->oMinX, ff->oMaxX, ff->oMinY, ff->oMaxY).y);
				}
				printf("拷贝的RCA大小%d\n", reSortCoordArray.size());

				CoordArray.push_back(reSortCoordArray[0]);
				CoordArray.push_back(reSortCoordArray[reSortCoordArray.size()-1]);
				for (size_t i = 1;i < reSortCoordArray.size()-1;i++)
				{
					CoordArray.push_back(reSortCoordArray[i]);


				}
			}
			printf("拷贝的CA大小%d\n",CoordArray.size());
			update();
		}
		void copyFromField(std::vector<Ramp::OriginalCoord> coord01)
		{
			auto ff = field->getDataPtr();
			if (coord01.size())
			{
				for (auto it : coord01)
				{
					MyCoord s;
					s.set(it.x, it.y);
					CoordArray.push_back(s);
					//printf("RCApush值：%d\n", ZeroOneToCoord(s, ff->oMinX, ff->oMaxX, ff->oMinY, ff->oMaxY).x, ZeroOneToCoord(s, ff->oMinX, ff->oMaxX, ff->oMinY, ff->oMaxY).y);
				}
				
			}
			printf("拷贝的CA大小%d\n", CoordArray.size());
			update();
		}

	public slots:
		void changeValue(int s);

	protected:
		void paintEvent(QPaintEvent* event);
		void mouseMoveEvent(QMouseEvent* event) override;
		void mousePressEvent(QMouseEvent* event)override;
		void mouseReleaseEvent(QMouseEvent* event)override;
		void reSort(std::vector<MyCoord>& vector1);
		void initializeLine(Dir mode );
		void updateFloatCoordArray() { floatCoord.clear(); for (auto it : reSortCoordArray) { floatCoord.push_back(CoordTo0_1Value(it)); } }
		void CoordtoField(std::vector<Coord0_1> CA, FVar<Ramp>* f) 
		{	
			auto s = f->getDataPtr();
			s->clearMyCoord();
			for (auto it : CA) 
			{
				s->addItemMyCoord(it.x,it.y);

			}
			for (auto it : CoordArray)
			{
				s->addItemOriginalCoord(it.x, it.y);
			}
			s->setOriginalCoord(minX,maxX,minY,maxY);
		}
		Coord0_1 CoordTo0_1Value(MyCoord& coord) 
		{//曲线坐标转换到0-1浮点值，并反转Y轴
			Coord0_1 s;
			float x = float(coord.x);
			float y = float(coord.y);
			float fmaxX = float(maxX);
			float fminX = float(minX);
			float fmaxY = float(maxY);
			float fminY = float(minY);

			s.x = (x - fminX) / (fmaxX - fminX);
			s.y = 1 - (y - fminY) / (fmaxY - fminY);
			return s;
		}
		MyCoord ZeroOneToCoord(Coord0_1& value, int x0,int x1,int y0, int y1) 
		{//0-1浮点值到曲线坐标
			MyCoord s;
			printf("x0: %d x1: %d y0: %d y1: %d\n", x0, x1, y0, y1);
			s.x = int(value.x * float(x1 - x0)) + x0;
			s.y = int((1-value.y) * float(y1 - y0)) + y0;
			std::cout << value.x << " --- " << value.y << std::endl;
			std::cout << s.x << " *** " << s.y << std::endl;
			return s;
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
		
		std::vector<MyCoord> CoordArray;
		std::vector<MyCoord> reSortCoordArray;
		MyCoord hoverCoord;
		MyCoord selectCoord;
		std::vector<Coord0_1> floatCoord;

	private:
		int w0;
		int h0;
		int currentW;
		int currentH;
		Dir Mode = x;
		BorderMode borderMode = Close;
		int hoverPoint = -1;
		int selectPoint = -1;
		bool isSelect = false;
		bool isHover = false;
		int maxX = 0;
		int maxY = 0;
		int minX = 0;
		int minY = 0;
		int selectDistance = 10;
		int radius = 4;
		FVar<Ramp>* field;

	};



}
