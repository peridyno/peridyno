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

		virtual ~mDrawLabel() {};



	};


}
