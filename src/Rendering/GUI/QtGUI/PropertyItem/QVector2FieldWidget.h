/**
 * Program:   Qt-based widget to visualize Vec2f or Vec2d
 * Module:    QVector2FieldWidget.h
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
#include "QFieldWidget.h"
#include "QtGUI/PPropertyWidget.h"
#include "Vector/Vector2D.h"

namespace dyno
{
	class QVector2FieldWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QVector2FieldWidget(FBase* field);
		QVector2FieldWidget(QString name, Vec2f v);
		~QVector2FieldWidget() override;

	signals:
		void vec2fChange(double, double);

	public slots:
		//Called when the widget is updated
		void updateField(double);

		//Called when the field is updated
		void updateWidget();

		void vec2fValueChange(double);

	private:
		QPiecewiseDoubleSpinBox* spinner1;
		QPiecewiseDoubleSpinBox* spinner2;

		Vec2f value;// active in "QVector2FieldWidget(QString name, Vec2f* v);"
	};

}
