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
#include "FCallbackFunc.h"
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

		QPiecewiseDoubleSpinBox(Real v, QWidget* parent = nullptr);

		double getRealValue() 
		{
			return realValue;
		}

		
		QLineEdit* getLineEdit() 
		{
			return this->lineEdit();
		}
		
		QValueDialog* mValueDialog = nullptr;



	private:
		//Prohibited to use
		void wheelEvent(QWheelEvent* event);

		void mousePressEvent(QMouseEvent* event) override;

		void mouseReleaseEvent(QMouseEvent* event) override;

		void mouseMoveEvent(QMouseEvent* event) override;

		void contextMenuEvent(QContextMenuEvent* event) override;



	protected:

		void stepBy(int steps) override
		{
			double val = realValue;
			double step = singleStep();

			double newVal = val + steps * step;

			if (newVal > maximum())
				newVal = maximum();
			else if (newVal < minimum())
				newVal = minimum();

			this->setValue(newVal);
			emit editingFinished();
		}

		QValidator::State validate(QString& input, int& pos) const override
		{
			Q_UNUSED(pos);

			if (input.isEmpty() || input == "-" || input == "+" || input == "." || input == "-." || input == "+.")
			{
				return QValidator::Intermediate;
			}

			bool ok = false;
			double val = input.toDouble(&ok);
			if (!ok)
			{
				return QValidator::Invalid;
			}

			if (val < minimum() || val > maximum())
			{
				return QValidator::Intermediate;
			}

			return QValidator::Acceptable;
		}

		void fixup(QString& input) const override
		{
			bool ok = false;
			double val = input.toDouble(&ok);
			if (!ok)
				return;

			if (val < minimum())
				input = QString::number(minimum(), 'g', decimalsMax);
			else if (val > maximum())
				input = QString::number(maximum(), 'g', decimalsMax);
		}

		virtual QString textFromValue(double val) const override
		{
			auto qstr = QString::number(realValue, 10, displayDecimals);

			return qstr;
		}
		
		void focusOutEvent(QFocusEvent* event) override
		{
			interpretText(); 
			onEditingFinished();
			QDoubleSpinBox::focusOutEvent(event);
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


		bool eventFilter(QObject* obj, QEvent* event) override;

		void createValueDialog();

	public:

	signals:

		void editingFinishedWithValue(double value);

	public:
		double setRealValue(double val);
		void setDouble(bool v) { isDouble = v; }
	public slots:


		void onEditingFinished()
		{
			
			if (fabs(this->value() - this->realValue) > (isDouble ? DBL_EPSILON : FLT_EPSILON))
			{
				this->setRealValue(this->value());
				emit editingFinishedWithValue(this->realValue);

			}
		}

		void triggerEditingFinished(double value)
		{
			this->setValue(value);
			this->interpretText(); 
			this->setRealValue(value);
			emit editingFinished();

		}

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

		bool isDouble = false;
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
		explicit QToggleLabel(std::string text,QWidget* parent = nullptr)
			: QLabel(parent)
		{
			this->setText(QString(text.c_str()));
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


	class mVec3fWidget : public QWidget
	{
		Q_OBJECT
	public:

		explicit mVec3fWidget(Vec3f v,std::string name,QWidget* parent = nullptr)
		{
			this->setContentsMargins(0, 0, 0, 0);
			QHBoxLayout* layout = new QHBoxLayout;

			nameLabel = new QToggleLabel(name.c_str());

			nameLabel->setMinimumWidth(90);

			this->setLayout(layout);

			v0 = new QPiecewiseDoubleSpinBox(parent);
			v1 = new QPiecewiseDoubleSpinBox(parent);
			v2 = new QPiecewiseDoubleSpinBox(parent);

			setRange(-999999,999999);

			v0->setValue(v[0]);
			v1->setValue(v[1]);
			v2->setValue(v[2]);

			v0->setMinimumWidth(90);
			v1->setMinimumWidth(90);
			v2->setMinimumWidth(90);

			layout->addWidget(nameLabel);
			layout->addWidget(v0);
			layout->addWidget(v1);
			layout->addWidget(v2);

			QObject::connect(v0, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double value) {emit vec3fChange(); });
			QObject::connect(v1, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double value) {emit vec3fChange(); });
			QObject::connect(v2, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double value) {emit vec3fChange(); });

			QObject::connect(nameLabel, SIGNAL(toggle(bool)), v0, SLOT(toggleDecimals(bool)));
			QObject::connect(nameLabel, SIGNAL(toggle(bool)), v1, SLOT(toggleDecimals(bool)));
			QObject::connect(nameLabel, SIGNAL(toggle(bool)), v2, SLOT(toggleDecimals(bool)));
		}

		~mVec3fWidget()
		{
			delete v0;
			delete v1;
			delete v2;
		}

		Vec3f getValue() 
		{
			return Vec3f(v0->getRealValue(), v1->getRealValue(), v2->getRealValue());
		};

		void setLabelMinimumWidth(int max)
		{
			nameLabel->setMinimumWidth(max);
		}

		void setRange(double min,double max) 
		{
			v0->setRange(min, max);
			v1->setRange(min, max);
			v2->setRange(min, max);
		}

		void setRange(double min0, double max0, double min1, double max1, double min2, double max2)
		{
			v0->setRange(min0, max0);
			v1->setRange(min1, max1);
			v2->setRange(min2, max2);
		}

	Q_SIGNALS:

		void vec3fChange();

	protected:

	private:

		QPiecewiseDoubleSpinBox* v0 = NULL;
		QPiecewiseDoubleSpinBox* v1 = NULL;
		QPiecewiseDoubleSpinBox* v2 = NULL;

		QToggleLabel* nameLabel;

	};



	class mPiecewiseDoubleSpinBox : public QWidget
	{
		Q_OBJECT
	public:

		explicit mPiecewiseDoubleSpinBox(double value, std::string name, QWidget* parent = nullptr)
		{
			this->setContentsMargins(0, 0, 0, 0);
			QHBoxLayout* layout = new QHBoxLayout;

			nameLabel = new QToggleLabel(name.c_str());
			nameLabel->setMinimumWidth(90);

			this->setLayout(layout);

			spinBox = new QPiecewiseDoubleSpinBox(parent);
			spinBox->setValue(value);

			spinBox->setRange(-99999999, 99999999);

			layout->addWidget(nameLabel);
			layout->addWidget(spinBox);

			QObject::connect(spinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double value) {emit valueChange(); });

			QObject::connect(nameLabel, SIGNAL(toggle(bool)), spinBox, SLOT(toggleDecimals(bool)));

		}

		~mPiecewiseDoubleSpinBox()
		{
			delete spinBox;
			delete nameLabel;
		}

		double getValue()
		{
			return spinBox->getRealValue();
		};

		void setLabelMinimumWidth(int max)
		{
			nameLabel->setMinimumWidth(max);
		}

		void setRange(double min,double max) 
		{
			spinBox->setRange(min,max);
		}

	Q_SIGNALS:

		void valueChange();

	protected:

	private:

		QPiecewiseDoubleSpinBox* spinBox = NULL;

		QToggleLabel* nameLabel = NULL;
	};



	class QVec3fWidget : public QWidget
	{
		Q_OBJECT
	public:

		explicit QVec3fWidget(Vec3f v, QWidget* parent = nullptr)
		{
			this->setContentsMargins(0, 0, 0, 0);
			QHBoxLayout* layout = new QHBoxLayout;

			this->setLayout(layout);

			v0 = new QPiecewiseDoubleSpinBox(parent);
			v1 = new QPiecewiseDoubleSpinBox(parent);
			v2 = new QPiecewiseDoubleSpinBox(parent);

			v0->setValue(v[0]);
			v1->setValue(v[1]);
			v2->setValue(v[2]);


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
