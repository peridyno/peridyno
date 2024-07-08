#include "QVehicleInfoWidget.h"

#include <QGridLayout>

#include "Field.h"
#include "QPiecewiseSpinBox.h"

#include <QPushButton.h>


namespace dyno
{
	IMPL_FIELD_WIDGET(VehicleBind, QVehicleInfoWidget)

		QVehicleInfoWidget::QVehicleInfoWidget(FBase* field)
		: QFieldWidget(field)
	{
		mainLayout = new QVBoxLayout;
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setAlignment(Qt::AlignLeft);

		this->setLayout(mainLayout);

		//Label

		auto titleLayout = new QVBoxLayout;

		QLabel* name = new QLabel();
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		titleLayout->addWidget(name);
		mainLayout->addLayout(titleLayout);

		//RigidBody UI
		auto RigidBodyUI = new QVBoxLayout;
		RigidBodyUI->setContentsMargins(0, 0, 0, 0);

		QHBoxLayout* nameLayout = new QHBoxLayout;

		QLabel* idLabel = new QLabel("<b>No.</b>");
		QLabel* rigidNameLabel = new QLabel("<b>Name</b>");
		//QLabel* rigidIdLabel = new QLabel("<b>RigidID</b>");
		QLabel* shapeIdLabel = new QLabel("<b>ShapeID</b>");
		QLabel* typeLabel = new QLabel("<b>Type</b>");
		QLabel* offsetLabel = new QLabel("<b>Offset</b>");

		QPushButton* addItembutton = new QPushButton("Add Item");
		addItembutton->setFixedSize(80, 30);

		nameLayout->addWidget(idLabel);
		nameLayout->addWidget(rigidNameLabel);
		//nameLayout->addWidget(rigidIdLabel);
		nameLayout->addWidget(shapeIdLabel);
		nameLayout->addWidget(typeLabel);
		nameLayout->addWidget(offsetLabel);
		nameLayout->addWidget(addItembutton);

		RigidBodyUI->addLayout(nameLayout);
		mainLayout->addLayout(RigidBodyUI);

		rigidsLayout = new QVBoxLayout;
		mainLayout->addLayout(rigidsLayout);

		idLabel->setFixedWidth(25);
		rigidNameLabel->setFixedWidth(100);
		typeLabel->setFixedWidth(76);
		//rigidIdLabel->setFixedWidth(76);
		shapeIdLabel->setFixedWidth(76);
		offsetLabel->setFixedWidth(76);

		idLabel->setAlignment(Qt::AlignCenter);
		rigidNameLabel->setAlignment(Qt::AlignCenter);
		typeLabel->setAlignment(Qt::AlignCenter);
		//rigidIdLabel->setAlignment(Qt::AlignCenter);
		shapeIdLabel->setAlignment(Qt::AlignCenter);
		offsetLabel->setAlignment(Qt::AlignCenter);

		QObject::connect(addItembutton, SIGNAL(pressed()), this, SLOT(addItemWidget()));
		QObject::connect(addItembutton, SIGNAL(pressed()), this, SLOT(updateJointComboBox()));

		//Joint UI
		auto jointUI = new QVBoxLayout;
		jointUI->setContentsMargins(0, 0, 0, 0);
		QHBoxLayout* jointLayout = new QHBoxLayout;

		QLabel* jointNumLabel = new QLabel("<b>No.</b>");
		QLabel* actor1 = new QLabel("<b>RigidBody1</b>");
		QLabel* actor2 = new QLabel("<b>RigidBody2</b>");
		QLabel* jointTypeLabel = new QLabel("<b>Type</b>");
		QLabel* moterLabel = new QLabel("<b>Moter</b>");
		QLabel* anchorOffsetLabel = new QLabel("<b>Anchor</b>");
		QLabel* rangeLabel = new QLabel("<b>Range</b>");
		QLabel* axisLabel = new QLabel("<b>Axis</b>");


		QPushButton* addJointItembutton = new QPushButton("Add Item");
		addJointItembutton->setFixedSize(80, 30);


		jointLayout->addWidget(jointNumLabel);
		jointLayout->addWidget(actor1);
		jointLayout->addWidget(actor2);
		jointLayout->addWidget(jointTypeLabel);
		jointLayout->addWidget(moterLabel);
		jointLayout->addWidget(anchorOffsetLabel);
		jointLayout->addWidget(rangeLabel);
		jointLayout->addWidget(axisLabel);
		jointLayout->addWidget(addJointItembutton);

		jointUI->addLayout(jointLayout);
		jointUI->setContentsMargins(0, 0, 0, 0);
		//jointUI->setAlignment(Qt::AlignLeft);
		mainLayout->addLayout(jointUI);

		jointsLayout = new QVBoxLayout;
		mainLayout->addLayout(jointsLayout);

		jointNumLabel->setFixedWidth(25);
		actor1->setFixedWidth(90);
		actor2->setFixedWidth(90);
		jointTypeLabel->setFixedWidth(65);
		moterLabel->setFixedWidth(65);
		anchorOffsetLabel->setFixedWidth(50);
		rangeLabel->setFixedWidth(45);
		axisLabel->setFixedWidth(45);

		jointNumLabel->setAlignment(Qt::AlignCenter);
		actor1->setAlignment(Qt::AlignCenter);
		actor2->setAlignment(Qt::AlignCenter);
		jointTypeLabel->setAlignment(Qt::AlignCenter);
		moterLabel->setAlignment(Qt::AlignCenter);
		anchorOffsetLabel->setAlignment(Qt::AlignCenter);
		rangeLabel->setAlignment(Qt::AlignCenter);
		axisLabel->setAlignment(Qt::AlignCenter);


		QObject::connect(addJointItembutton, SIGNAL(pressed()), this, SLOT(addJointItemWidget()));
		//QObject::connect(addJointItembutton, SIGNAL(pressed()), this, SLOT(updateJointComboBox()));



		QObject::connect(this, SIGNAL(vectorChange()), this, SLOT(updateField()));

		FVar<VehicleBind>* f = TypeInfo::cast<FVar<VehicleBind>>(field);
		if (f != nullptr)
		{
			mVec = f->getValue();
		}


		updateWidget();

	};

	void QVehicleInfoWidget::updateWidget()
	{
		for (size_t i = 0; i < mVec.vehicleRigidBodyInfo.size(); i++)
		{
			createItemWidget(mVec.vehicleRigidBodyInfo[i]);
		}

		buildMap();

		for (size_t i = 0; i < mVec.vehicleJointInfo.size(); i++)
		{
			createJointItemWidget(mVec.vehicleJointInfo[i]);
		}
		updateJointComboBox();
		updateVector();
	}

}

