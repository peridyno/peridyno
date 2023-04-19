/**
 * Program:   Qt-based widget to visualize Vec3i or Vec3u
 * Module:    QVector3iFieldWidget.h
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
#include "QFieldWidget.h"

namespace dyno
{
	class QVector3iFieldWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		QVector3iFieldWidget(FBase* field);
		~QVector3iFieldWidget() override;

	public slots:
		//Called when the field is updated
		void updateField(int);

		//Called when the widget is updated
		void updateWidget();

	private:
		QSpinBox* spinner1;
		QSpinBox* spinner2;
		QSpinBox* spinner3;
	};
}
