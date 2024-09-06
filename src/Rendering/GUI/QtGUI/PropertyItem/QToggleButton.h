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
#include "QFieldWidget.h"
#include <QKeyEvent>
#include <QEvent>

//PeriDyno
#include "Field.h"
#include "Format.h"
#include "FCallBackFunc.h"

//C++
#include <memory>

namespace dyno
{
	class Node;
	class Module;
	class FBase;

	class QDoubleSpinner;
	class QDoubleSlider;

	

	class QToggleButton : public QPushButton
	{
		Q_OBJECT

	public:

		QToggleButton(QWidget* pParent = NULL);

		QToggleButton(bool isChecked, QWidget* pParent = NULL);

		void setText(std::string textUnCheck, std::string textCheck);

		void setValue(bool press) 
		{ 
			isPress = press; 		
			updateText();
		}

		void updateText() 
		{
			QString t;
			if (isPress)
			{
				t = QString::fromStdString(textChecked);
			}
			else
			{
				t = QString::fromStdString(textUnChecked);
			}
			this->QPushButton::setText(t);
		}

	Q_SIGNALS:
		void clicked();

	public slots:
		void ModifyText();

	public:

		bool isPress = false;

	private:
		
		std::shared_ptr<FCallBackFunc> callback = nullptr;
		std::string textUnChecked = "unCheck";
		std::string textChecked = "Check";//

	};


}
