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

//C++
#include <memory>

namespace dyno
{
	class Node;
	class Module;
	class FBase;

	class QDoubleSpinner;
	class QDoubleSlider;
	class mDoubleSpinBox;
	class ValueDialog;



	class mDoubleSpinBox : public QDoubleSpinBox
	{
		Q_OBJECT
	public:
		explicit mDoubleSpinBox(QWidget* parent = nullptr);

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
		
		ValueDialog* ValueModify = nullptr;

		bool ModifyByDialog;

	private:
		//Prohibited to use
		void wheelEvent(QWheelEvent* event);

		void mousePressEvent(QMouseEvent* event) override;

		void mouseReleaseEvent(QMouseEvent* event) override;

		void mouseMoveEvent(QMouseEvent* event) override;
		
		void buildDialog();

		void contextMenuEvent(QContextMenuEvent* event) override;

	protected:
		

		virtual QString textFromValue(double val) const override 
		{
			return QString::number(realValue, 10, displayDecimals);
		}
			
	public:



	signals:
	public slots:
		void ModifyValue(double);

		void LineEditFinished(double);
		void LineEditStart();
		
		void toggleDecimals(bool v) 
		{
			if (v)
				displayDecimals = decimalsMax;
			else
				displayDecimals = decimalsMin;

			this->lineEdit()->setText(QString::number(realValue, 10, displayDecimals));
		}
		
	private:
		int decimalsMin = 3;
		int decimalsMax = 8;
		int displayDecimals = 3;
		double realValue = 0;
	};


	class toggleLabel : public QLabel
	{
		Q_OBJECT
	public:

		explicit toggleLabel(QWidget* parent = nullptr)
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


}
