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

		void fieldUpdated();

	private:
		FBase* mField = nullptr;

		QDoubleSlider* slider = nullptr;

		std::shared_ptr<FCallBackFunc> callback = nullptr;
	};


	class mDoubleSpinBox : public QDoubleSpinBox
	{
		Q_OBJECT
	public:
		explicit mDoubleSpinBox(QWidget* parent = nullptr);
	private:
		//Prohibited to use
		void wheelEvent(QWheelEvent* event);
	signals:
	public slots:
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

		void fieldUpdated();

	private:
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

		void fieldUpdated();

	private:
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
}
