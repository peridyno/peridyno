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

#include "Field/Ramp.h"
#include "PCustomWidgets.h"
#include "QBoolFieldWidget.h"
#include "qcheckbox.h"
#include <algorithm>
#include <QPainterPath>
#include "qlayout.h"
#include "Field.h"
#include "QComponent.h"

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


	class QDrawLabel : public mDrawLabel
	{
		Q_OBJECT
	public slots:
		void changeValue(int s);
		void changeInterpValue(int s);

	public:
		QDrawLabel(QWidget* parent = nullptr);
		~QDrawLabel();

		void updateDataToField()override;
		void setField(FVar<Ramp>* f) { mField = TypeInfo::cast<FVar<Ramp>>(f);}
		void copySettingFromField();

	protected:
		void paintEvent(QPaintEvent* event);
		void mouseMoveEvent(QMouseEvent* event) override;
		void mousePressEvent(QMouseEvent* event)override;
		int addPointtoEnd() override;
		void CoordtoField(Ramp& s);

	private:
		FVar<Ramp>* mField;
	};



}
