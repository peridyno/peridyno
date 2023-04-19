#include "QStateFieldWidget.h"

#include <QHBoxLayout>
#include <QCheckBox>

#include "Node.h"

namespace dyno
{
	QStateFieldWidget::QStateFieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		//this->setStyleSheet("border:none");
		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedHeight(24);
		name->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		layout->addWidget(name, 0);
		layout->addStretch(1);

		QCheckBox* checkbox = new QCheckBox();		
		checkbox->setFixedWidth(20);
		layout->addWidget(checkbox, 0);

		if (field->parent()->findOutputField(field))
		{
			checkbox->setChecked(true);;
		}
		else
		{
			checkbox->setChecked(false);
		}

		//TODO: use another way
		connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(tagAsOuput(int)));
	}

	void QStateFieldWidget::tagAsOuput(int status)
	{
		emit stateUpdated(field(), status);
	}
}

