#include "QVectorWidget.h"

#include <QGridLayout>

#include "Field.h"
#include "QPiecewiseSpinBox.h"

#include <QPushButton.h>

namespace dyno
{
	IMPL_FIELD_WIDGET(std::vector<int>, QVectorWidget)

	QVectorWidget::QVectorWidget(FBase* field)
		: QFieldWidget(field)
	{
		mainLayout = new QVBoxLayout;
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setAlignment(Qt::AlignLeft);

		this->setLayout(mainLayout);

		//Label
		QHBoxLayout* nameLayout = new QHBoxLayout;
		QLabel* name = new QLabel();
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		QPushButton* addItembutton = new QPushButton("add Item");
		addItembutton->setFixedSize(100, 40);

		nameLayout->addWidget(name);
		nameLayout->addWidget(addItembutton);

		mainLayout->addLayout(nameLayout);

		////Set default GUI style
		//QFile file(":/dyno/DarkStyle.qss");
		//file.open(QIODevice::ReadOnly);

		//QString style = file.readAll();
		//this->setStyleSheet(style);

		FVar<std::vector<int>>* f = TypeInfo::cast<FVar<std::vector<int>>>(field);
		if (f != nullptr) 
		{
			printf("vector int\n");
			mVec = f->getValue();
		}

		
		QObject::connect(addItembutton, SIGNAL(pressed()), this, SLOT(addItemWidget()));


	};



}

