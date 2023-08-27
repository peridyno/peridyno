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

	class PERIDYNO_QTGUI_API QFieldWidget : public QGroupBox
	{
		Q_OBJECT
	public:
		QFieldWidget(FBase* field);
		virtual ~QFieldWidget();

		inline FBase* field() { return mField; }

	signals:
		void fieldChanged();

	public slots:

	private:
		FBase* mField = nullptr;

	private:
		void syncValueFromField();

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
		explicit ValueDialog(QWidget* parent = nullptr) {};

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
}
