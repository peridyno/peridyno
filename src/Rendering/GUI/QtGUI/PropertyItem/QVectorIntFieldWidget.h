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

	class mVectorItemLayout : public QHBoxLayout
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET
		mVectorItemLayout(int id)
		{
			mId = id;

			index = new QLabel(std::to_string(id).c_str());
			this->addWidget(index, 0, 0);

			inputSpin = new QSpinBox;
			inputSpin->setValue(0);
			this->addWidget(inputSpin, 0, 0);

			removeButton = new QPushButton("Delete");
			this->addWidget(removeButton, 0, 0);

			//this->layout()->addWidget(index);
			QObject::connect(removeButton, SIGNAL(pressed()), this, SLOT(emitSignal()));
			QObject::connect(inputSpin, SIGNAL(valueChanged(int)), this, SLOT(emitChange(int)));


		};


		~mVectorItemLayout()
		{
			delete inputSpin;
			delete removeButton;
			delete index;
		};

		int value() { return inputSpin->value(); };
		void setValue(int v) { inputSpin->setValue(v); }
		void setId(int id) { mId = id; index->setText(std::to_string(id).c_str()); };

	signals:
		void removeById(int);
		void valueChange(int);


	public slots:
		void emitSignal()
		{
			emit removeById(mId);
		}
		void emitChange(int v)
		{
			emit valueChange(v);
		}



	private:
		int mId = -1;
		QSpinBox* inputSpin = nullptr;
		QPushButton* removeButton = nullptr;
		QLabel* index = nullptr;
	};


	class QVectorIntFieldWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QVectorIntFieldWidget(FBase* field);

		~QVectorIntFieldWidget() override {};

	signals:
		void vectorChange();


	public slots:
		//Called when the widget is updated

		void updateField()
		{
			FVar<std::vector<int>>* f = TypeInfo::cast<FVar<std::vector<int>>>(field());
			if (f != nullptr)
			{
				f->setValue(mVec);
			}
		};

		//Called when the field is updated
		void updateWidget();



		void updateVector(int) { updateVector(); }

		void updateVector()
		{
			mVec.clear();
			for (size_t i = 0; i < allItem.size(); i++)
			{
				mVec.push_back(allItem[i]->value());
			}
			emit vectorChange();
			//printField();
		}

		void addItemWidget()
		{
			mVectorItemLayout* itemLayout = new mVectorItemLayout(allItem.size());
			QObject::connect(itemLayout, SIGNAL(removeById(int)), this, SLOT(removeItemWidgetById(int)));
			QObject::connect(itemLayout, SIGNAL(valueChange(int)), this, SLOT(updateVector()));

			mainLayout->addLayout(itemLayout);
			allItem.push_back(itemLayout);

			updateVector();
		}


		void removeItemWidgetById(int id)
		{
			mainLayout->removeItem(allItem[id]);
			delete allItem[id];
			allItem.erase(allItem.begin() + id);
			for (size_t i = 0; i < allItem.size(); i++)
			{
				allItem[i]->setId(i);
			}

			mainLayout->update();

			updateVector();
		}


	private:


		void createItemWidget(int v)
		{
			{
				mVectorItemLayout* itemLayout = new mVectorItemLayout(allItem.size());
				itemLayout->setValue(v);

				QObject::connect(itemLayout, SIGNAL(removeById(int)), this, SLOT(removeItemWidgetById(int)));
				QObject::connect(itemLayout, SIGNAL(valueChange(int)), this, SLOT(updateVector()));

				mainLayout->addLayout(itemLayout);
				allItem.push_back(itemLayout);
			};
		}

		void printField()
		{
			for (size_t i = 0; i < mVec.size(); i++)
			{
				std::cout << mVec[i] << ", ";
			}
			std::cout << std::endl;
		}




	private:

		std::vector<int> mVec;


		QVBoxLayout* mainLayout = nullptr;

		std::vector<mVectorItemLayout*> allItem;


	};



}
