/**
 * Program:   Qt-based widget to visualize a float or double type value
 * Module:    QRealFieldWidget.h
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
#include "PCustomWidgets.h"
#include "QtGUI/PPropertyWidget.h"

namespace dyno
{
	class QRealFieldWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QRealFieldWidget(FBase* field);
		~QRealFieldWidget() override;

	public slots:
		//Called when the field is updated
		void updateWidget();

		//Called when the widget is updated
		void updateField(double);

	private slots:
		void onSliderValueChanged(double val);

	private:
		QDoubleSlider* slider = nullptr;
		QPiecewiseDoubleSpinBox* spinner = nullptr;
	};
}
