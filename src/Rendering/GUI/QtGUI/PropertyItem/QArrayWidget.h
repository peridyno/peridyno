/**
 * Program:  
 * Module:    
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
#include <QWidget>
#include <QVBoxLayout>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <vector>
#include "QVector3FieldWidget.h"
#include <typeinfo>
#include <memory>
#include "LockerButton.h"

namespace dyno
{

    class ArrayWidgetBase : public QFieldWidget
    {
        Q_OBJECT
    public:

        ArrayWidgetBase(FBase* t);
        
        void connectSignal(QFieldWidget* field, int id);

        virtual void updataData(int id) = 0;
        virtual void rebuildUI() = 0;
        virtual void updateArray() = 0;

        virtual void addItem(QWidget* item) = 0;

    public slots:
        void OnElementUpdate(int id) 
        {
            updataData(id);
        };

    protected:
        bool mFlag = true;
        QVBoxLayout* layout = NULL;
        QLabel* sizeLabel = NULL;
        LockerButton* addButton = NULL;
        LockerButton* deleteButton = NULL;

    };

    template<typename T>
    class ArrayWidget : public ArrayWidgetBase//: public QGroupBox
    {
    public:
     
        explicit ArrayWidget(FBase* t);

        void addItem(QWidget* item) override;

    private:
        FCArray<T>* dataPtr;
        std::vector<FVar<T>*> fVarArray;
        
        std::vector<QFieldWidget*> widgets;
        virtual void rebuildUI()override;
        virtual void updateArray()override;
        virtual void updataData(int id)override 
        {
            (*dataPtr->getDataPtr())[id] = fVarArray[id]->getValue();
        }
    };

    template<typename T>
    ArrayWidget<T>::ArrayWidget(FBase* t)
        : ArrayWidgetBase(t)
    {
        auto farray = dynamic_cast<FCArray<T>*> (t);
        if (farray)
        {
            dataPtr = farray;
            for (size_t i = 0; i < dataPtr->size(); i++)
                fVarArray.push_back(new FVar<T>((*dataPtr->constDataPtr())[i], "var_[Element " + std::to_string(i) + "]", "", FieldTypeEnum::Param, NULL));

            //rebuildUI();
        }       
    }

	template<typename T>
	void ArrayWidget<T>::addItem(QWidget* item)
	{
		//if (QFieldWidget* field = qobject_cast<QFieldWidget*>(item))
		{
			//isValidWidgetType = true;

//             connectSignal(field, widgets.size());
// 			widgets.push_back(field);

			QHBoxLayout* elementLayout = new QHBoxLayout;
			QLabel* label = new QLabel;
			label->setFixedWidth(30);
			elementLayout->addWidget(label);
			elementLayout->addWidget(item);
			layout->addLayout(elementLayout);
		}
	}

    template<typename T>
    void ArrayWidget<T>::rebuildUI()
    {
        for (size_t i = 0; i < widgets.size(); i++)
        {
            widgets[i]->disconnect();
        }
        widgets.clear();

        bool isValidWidgetType = false;

        for (size_t i = 0; i < fVarArray.size(); i++)
        {
            FVar<T>* it = fVarArray[i];
            auto item = PPropertyWidget::createFieldWidget(it);

            this->addItem(item);
        }
//         if (!isValidWidgetType) 
//         {
//             layout->addWidget(new QLabel(std::string("Data Size : " + std::to_string(dataPtr->size())).c_str()));
//             layout->addWidget(new QLabel(std::string("UnsupportedFieldWidget : " + dataPtr->getTemplateName()).c_str()));
//         }
    }

    template<typename T>
    void ArrayWidget<T>::updateArray()
    {
        dataPtr->resize(fVarArray.size());

        for (size_t i = 0; i < fVarArray.size(); i++)
        {
            FVar<T>* item = fVarArray[i];
            (*dataPtr->constDataPtr())[i] = item->getValue();
        }
        dataPtr->getDataPtr();

    }
}
