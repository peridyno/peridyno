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

#include "Field/Curve.h"
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

	public:
		QCurveLabel(QWidget* parent = nullptr);
		QCurveLabel(FVar<Curve>* f, QWidget* parent = nullptr);
		~QCurveLabel();

		void updateDataToField()override;
		void setField(FVar<Curve>* f){ mField = TypeInfo::cast<FVar<Curve>>(f);}
		void copySettingFromField();

	protected:
		void paintEvent(QPaintEvent* event);
		void mouseMoveEvent(QMouseEvent* event) override;
		void mousePressEvent(QMouseEvent* event)override;
		int addPointtoEnd()override;
		void CoordtoField(Curve& s);

	private:
		FVar<Curve>* mField;

	};



}
