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
#include <QVBoxLayout>

//PeriDyno
#include "Format.h"
#include "FCallbackFunc.h"
#include "QtGUI/Common.h"

//C++
#include <memory>
#include <sstream>

namespace dyno
{
	class Node;
	class Module;
	class FBase;

	class QDoubleSpinner;
	class QDoubleSlider;
	class QPiecewiseDoubleSpinBox;
	class QValueDialog;

	/**
	 * A piecewise spin box
	 */
	class QPiecewiseSpinBox : public QSpinBox
	{
		Q_OBJECT
	public:
		explicit QPiecewiseSpinBox(QWidget* parent = nullptr);

	public:

	protected:
		bool eventFilter(QObject* obj, QEvent* event) override;

		void createValueDialog();

		QValidator::State validate(QString& input, int& pos) const override
		{
			if (input.isEmpty())
				return QValidator::Intermediate;

			bool ok = false;
			int val = input.toInt(&ok);

			if (!ok)
				return QValidator::Invalid;

			if (val < minimum())
			{
				input = QString::number(this->minimum());
			}
			else if (val > maximum())
			{
				input = QString::number(this->maximum());
			}

			return QValidator::Acceptable;
		}

	private:
		
		void wheelEvent(QWheelEvent* event);

		void contextMenuEvent(QContextMenuEvent* event) override;

		void mousePressEvent(QMouseEvent* event) override;

	private:
		QValueDialog* mValueDialog = nullptr;
	};
}
