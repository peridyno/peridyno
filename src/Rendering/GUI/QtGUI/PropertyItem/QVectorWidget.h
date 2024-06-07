/**
 * Program:   Qt-based widget to visualize Vec3f or Vec3d
 * Module:    QVector3FieldWidget.h
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
#include "Field.h"
#include <QHBoxLayout>
#include <QVBoxLayout>

namespace dyno
{
	class QVectorWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QVectorWidget(FBase* field);

		~QVectorWidget() override {};

	signals:
		void vectorChange(std::vector<int> vec);

	public slots:
		//Called when the widget is updated

		void updateField(std::vector<int> vec) 
		{
			FVar<std::vector<int>>* f = TypeInfo::cast<FVar<std::vector<int>>>(field());
			if (f != nullptr)
			{
				f->setValue(mVec);
			}
		};

		//Called when the field is updated
		void updateWidget(std::vector<int> vec) {};

		void addItemWidget()
		{
			printf("press\n");

			QHBoxLayout* itemLayout = new QHBoxLayout;

			QLabel* index = new QLabel("id");
			itemLayout->addWidget(index, 0, 0);

			QSpinBox* inputWidget = new QSpinBox;
			inputWidget->setValue(0);
			itemLayout->addWidget(inputWidget, 0, 0);

			QPushButton* removeButton = new QPushButton("*");
			itemLayout->addWidget(removeButton, 0, 0);

			mainLayout->addLayout(itemLayout);

			//this->layout()->addWidget(index);
			QObject::connect(removeButton, SIGNAL(pressed()), this, SLOT(addItemWidget()));

			
		}




	private:

		void addItem(int element)
		{
			mVec.push_back(element);
			emit vectorChange(mVec);
		}

		void removeItem(int id)
		{
			if (id >= mVec.size())
				return;
			mVec.erase(mVec.begin() + id);
			emit vectorChange(mVec);
		}
		




	private:

		std::vector<int> mVec;
		QVBoxLayout* mainLayout = nullptr;
	};



}
