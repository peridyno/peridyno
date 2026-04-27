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
#include "Field/FList.h"
#include <QLabel>
#include <QPushButton>

namespace dyno
{
    class ArrayWidget : public QFieldWidget
    {
        Q_OBJECT
    public:

        ArrayWidget(FBase* t);

        virtual void addItem(QWidget* item, int id)
        {
            if (!item)
                return;

            QHBoxLayout* elementLayout = new QHBoxLayout;
            elementLayout->setAlignment(Qt::AlignVCenter);
            QLabel* label = new QLabel;
            label->setFixedWidth(30);
            elementLayout->addWidget(label);
            elementLayout->addWidget(item);
            listLayout->addLayout(elementLayout);

            auto deleteButton = new QPushButton("-");
            elementLayout->addWidget(deleteButton);
            deleteButton->setFixedSize(20,20);
            deleteButton->setStyleSheet(R"(
                QPushButton {
                    background-color: #464646;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #616161;
                }
                QPushButton:pressed {
                    background-color: #000000;
                }
            )");

            connect(deleteButton, &QPushButton::clicked, [this, elementLayout]() {
                int index = listLayout->indexOf(elementLayout);
                this->deleteElement(index);
            });
        };

    signals:
        void onRebuildElement();
        void onAddElement();

    public slots:
        void deleteElement(int id)
        {
            removeSubLayout(id, listLayout);
            emit onDeleteElement(id);
        }

        void clearList()
        {
            if (listLayout)
                clearLayoutRecursively(listLayout);

            printf("ListWidget::Clear list completed\n");
        }

        void addElement()
        {
            printf("ListWidget::Need to implement the add element function\n");
            emit onAddElement();
        }

        void onDeleteElement(int id)
        {
            printf("ListWidget::Delete element at index: %d\n", id);
        }

    protected:
        bool mFlag = true;
        QVBoxLayout* listLayout = nullptr;
        QLabel* sizeLabel = nullptr;
        QPushButton* addButton = nullptr;
        QPushButton* clearButton = nullptr;

    private:
        void clearLayoutRecursively(QLayout* layout) {
            if (!layout) return;
            while (QLayoutItem* item = layout->takeAt(0)) {
                if (QLayout* childLayout = item->layout()) {
                    clearLayoutRecursively(childLayout);
                    delete childLayout;   // 递归删除子布局对象
                }
                else if (QWidget* widget = item->widget()) {
                    widget->deleteLater();
                }
                delete item;
            }
        }

        void removeSubLayout(int index, QVBoxLayout* parentLayout) {
            if (!parentLayout || index < 0 || index >= parentLayout->count())
                return;

            QLayoutItem* item = parentLayout->takeAt(index);
            if (!item) return;

            if (QLayout* subLayout = item->layout()) {
                clearLayoutRecursively(subLayout);
                subLayout->deleteLater();
                delete item;  
            }
            else {
                delete item;
            }
        }
    };
}