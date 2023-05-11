/**
 * Program:   Qt-based widget to visualize Color
 * Module:    QVector3iFieldWidget.h
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

//RenderCore
#include "Color.h"

namespace dyno
{
	class QColorButton : public QPushButton
	{
		Q_OBJECT

	public:
		QColorButton(QWidget* pParent = NULL);

		int		getMargin(void) const;
		void	setMargin(const int& Margin);
		int		getRadius(void) const;
		void	setRadius(const int& Radius);
		QColor	getColor(void) const;
		void	setColor(const QColor& Color, bool BlockSignals = false);

	protected:
		virtual void paintEvent(QPaintEvent* event);
		virtual void mousePressEvent(QMouseEvent* event);

	private slots:
		void onColorChanged(const QColor& Color);

	signals:
		void colorChanged(const QColor&);

	private:
		int		mMargin;
		int		mRadius;
		QColor	mColor;
	};


	class QColorWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		QColorWidget(FBase* field);
		~QColorWidget() override;

	public slots:
		//Called when the widget is updated
		void updateField(int);

		void updateColorWidget(const QColor& color);

	private:
		QSpinBox* spinner1;
		QSpinBox* spinner2;
		QSpinBox* spinner3;

		QColorButton* colorButton;
	};
}
