/**
 * Program:   Parent class for all field widgets
 * Module:    QFieldWidget.h
 *
 * Copyright 2023 Xiaowei He
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
//Qt
#include <QGroupBox>
#include <QPushButton>
#include <QSpinBox>
#include <QDialog>
#include <QLineEdit>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QLabel>
#include <QVBoxLayout>

//PeriDyno
#include "Format.h"
#include "FCallBackFunc.h"
#include "QtGUI/Common.h"
#include "Core/Vector.h"

//C++
#include <memory>
#include <sstream>

namespace dyno
{
	class Node;
	class Module;
	class FBase;

	class QDoubleSpinner;
	class QDoubleSlider;
	class QPiecewiseDoubleSpinBox;
	class QValueDialog;

	class QPiecewiseDoubleSpinBox : public QDoubleSpinBox
	{
		Q_OBJECT
	public:
		explicit QPiecewiseDoubleSpinBox(QWidget* parent = nullptr);

		double getRealValue() 
		{
			return realValue;
		}
		double setRealValue(double val)
		{
			realValue = val;
			return realValue;
		}
		
		QLineEdit* getLineEdit() 
		{
			return this->lineEdit();
		}
		
		QValueDialog* ValueModify = nullptr;


	private:
		//Prohibited to use
		void wheelEvent(QWheelEvent* event);

		void mousePressEvent(QMouseEvent* event) override;

		void mouseReleaseEvent(QMouseEvent* event) override;

		void mouseMoveEvent(QMouseEvent* event) override;

		void contextMenuEvent(QContextMenuEvent* event) override;

	protected:
		

		virtual QString textFromValue(double val) const override
		{
			auto qstr = QString::number(realValue, 10, displayDecimals);

			return qstr;
		}
		
		virtual double valueFromText(const QString& text) const override 
		{
			if (istoggle)
			{
				return realValue;
			}
			else 
			{
				return text.toDouble();
			}
		}

	public:

	signals:


	public slots:
		void ModifyValue(double);
		void ModifyValueAndUpdate(double);
		void LineEditFinished(double);
		void LineEditStart();
		
		void toggleDecimals(bool v) 
		{
			if (v)
				displayDecimals = decimalsMax;
			else
				displayDecimals = decimalsMin;

			istoggle = true;

			auto t = QString::number(realValue, 10, displayDecimals);
			this->lineEdit()->setText(t);

			istoggle = false;
		}
		
	private:
		int decimalsMin = 3;
		int decimalsMax = 8;
		int displayDecimals = 3;
		double realValue = 0;
		bool istoggle = false;
	};


	class QToggleLabel : public QLabel
	{
		Q_OBJECT
	public:

		explicit QToggleLabel(QWidget* parent = nullptr)
			: QLabel(parent)
		{

		}

	Q_SIGNALS:
		void toggle(bool high);

	protected:
		void mousePressEvent(QMouseEvent* event) override
		{
			current = !current;
			emit toggle(current);
		}


	private:

		bool current = false;
	};


	class QVec3fWidget : public QWidget
	{
		Q_OBJECT
	public:

		explicit QVec3fWidget(Vec3f v)
		{
			v0 = new QPiecewiseDoubleSpinBox;
			v1 = new QPiecewiseDoubleSpinBox;
			v2 = new QPiecewiseDoubleSpinBox;

			v0->setRealValue(v[0]);
			v1->setRealValue(v[1]);
			v2->setRealValue(v[2]);

			QHBoxLayout* layout = new QHBoxLayout;

			layout->addWidget(v0);
			layout->addWidget(v1);
			layout->addWidget(v2);

			QObject::connect(v0, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double value) {emit vec3fChange(); });
			QObject::connect(v1, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double value) {emit vec3fChange(); });
			QObject::connect(v2, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double value) {emit vec3fChange(); });
		}

		~QVec3fWidget() 
		{
			delete v0;
			delete v1;
			delete v2;
		}

		Vec3f getValue() 
		{
			return Vec3f(v0->getRealValue(), v1->getRealValue(), v2->getRealValue());
		};

	Q_SIGNALS:

		void vec3fChange();

	protected:

	private:

		QPiecewiseDoubleSpinBox* v0 = NULL;
		QPiecewiseDoubleSpinBox* v1 = NULL;
		QPiecewiseDoubleSpinBox* v2 = NULL;


	};

}
