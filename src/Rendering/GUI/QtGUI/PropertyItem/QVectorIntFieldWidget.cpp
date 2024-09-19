#include "QVectorIntFieldWidget.h"

#include <QGridLayout>

#include "Field.h"
#include "QPiecewiseSpinBox.h"

#include <QPushButton>

namespace dyno
{
	IMPL_FIELD_WIDGET(std::vector<int>, QVectorIntFieldWidget)

		QVectorIntFieldWidget::QVectorIntFieldWidget(FBase* field)
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


		QObject::connect(addItembutton, SIGNAL(pressed()), this, SLOT(addItemWidget()));
		QObject::connect(this, SIGNAL(vectorChange()), this, SLOT(updateField()));

		FVar<std::vector<int>>* f = TypeInfo::cast<FVar<std::vector<int>>>(field);
		if (f != nullptr)
		{
			mVec = f->getValue();
		}

		updateWidget();

	};

	void QVectorIntFieldWidget::updateWidget()
	{
		for (size_t i = 0; i < mVec.size(); i++)
		{
			createItemWidget(mVec[i]);
		}

	}

}

