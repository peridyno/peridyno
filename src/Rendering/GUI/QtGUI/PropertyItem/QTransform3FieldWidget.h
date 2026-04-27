/**
 * Program:   Qt-based widget to visualize Transform3f or Transform3d
 * Module:    QTransform3FieldWidget.h
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

namespace dyno
{
	class QTransform3FieldWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QTransform3FieldWidget(FBase* field);
		~QTransform3FieldWidget() override;

	public slots:
		//Called when the field is updated
		void updateField(double);

		void updateWidget();

	private:
		QPiecewiseDoubleSpinBox* transSpinners[3];
		QPiecewiseDoubleSpinBox* scaleSpinners[3];
		QPiecewiseDoubleSpinBox* rotSpinners[9];
		QPiecewiseDoubleSpinBox* rotAngleSpinners[3];
		QPushButton* rotModeToggle;
		bool useAngleMode; // true for angle input, false for matrix input
	};
}
