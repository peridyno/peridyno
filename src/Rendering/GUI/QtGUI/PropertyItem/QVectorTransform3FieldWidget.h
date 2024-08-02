/**
 * Program:   Qt-based widget to visualize Vec3f or Vec3d
 * Module:    QVector3FieldWidget.h
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
#include "Field.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include "QPiecewiseDoubleSpinBox.h"
#include "qgroupbox.h"

namespace dyno
{
	class mVectorTransformItemLayout : public QHBoxLayout
	{
		Q_OBJECT
	public:
		mVectorTransformItemLayout(int id);

		~mVectorTransformItemLayout();

		Transform3f value();

		void setValue(Transform3f v);

		void setId(int id) { mId = id; index->setText(std::to_string(id).c_str()); };

	signals:

		/**
		 * @brief Called When the RemoveButton is clicked.
		 */
		void removeById(int);

		/**
		 * @brief Called when the Widget changed.
		 */
		void valueChange(double);

	public slots:
		void emitSignal() { emit removeById(mId); }

		void emitChange(double v) { emit valueChange(v); }


	private:

	private:
		int mId = -1;

		QGroupBox* mGroup = NULL;

		QPiecewiseDoubleSpinBox* mT0 = NULL;
		QPiecewiseDoubleSpinBox* mT1 = NULL;
		QPiecewiseDoubleSpinBox* mT2 = NULL;

		QPiecewiseDoubleSpinBox* mR0 = NULL;
		QPiecewiseDoubleSpinBox* mR1 = NULL;
		QPiecewiseDoubleSpinBox* mR2 = NULL;

		QPiecewiseDoubleSpinBox* mS0 = NULL;
		QPiecewiseDoubleSpinBox* mS1 = NULL;
		QPiecewiseDoubleSpinBox* mS2 = NULL;

		QLabel* mTLabel = NULL;
		QLabel* mRLabel = NULL;
		QLabel* mSLabel = NULL;

		QPushButton* removeButton = nullptr;
		QLabel* index = nullptr;
	};

	class QVectorTransform3FieldWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QVectorTransform3FieldWidget(FBase* field);

		~QVectorTransform3FieldWidget() override {};

	signals:
		void vectorChange();

	public slots:
		/**
		 * @brief Called when the widget is updated.
		 */
		void updateField();
		/**
		 * @brief Called when the field is updated.
		 */
		void updateWidget();

		void updateVector(int) { updateVector(); }
		/**
		 * @brief Update "std::vector<Transform3f> mVec".
		 */
		void updateVector();
		/**
		 * @brief Called when the QPushButton* addItembutton is clicked.
		 */
		void addItemWidget();
		/**
		 * @brief Called when the "mVectorTransformItemLayout::QPushButton* removeButton" is clicked.
		 */
		void removeItemWidgetById(int id);

	private:
		/**
		 * @brief Creating Vector Elements(std::vector<mVectorTransformItemLayout*>) from fields.
		 */
		void createItemWidget(Transform3f v);


	private:

		std::vector<Transform3f> mVec;

		QVBoxLayout* mMainLayout = NULL;

		std::vector<mVectorTransformItemLayout*> mItems;

	};


	

}
