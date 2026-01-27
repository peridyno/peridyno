/**
 * Program:   Qt Property widgets to support string or file path
 * Module:    QFilePathWidget.h
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
	class QStringFieldWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QStringFieldWidget(FBase* field);
		~QStringFieldWidget() override {};

	public slots:
		void updateField(QString str);
		void updateWidget();
	private:
		QLineEdit* fieldname;
		FVar<std::string>* f;
	};

	class QFilePathWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QFilePathWidget(FBase* field);
		~QFilePathWidget() override {};

	public slots:
		void updateField(QString str);

	private:
		QLineEdit* location;
	};
}
