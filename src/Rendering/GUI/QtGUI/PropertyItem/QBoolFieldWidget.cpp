#include "QBoolFieldWidget.h"

#include <QHBoxLayout>
#include <QCheckBox>
#include <QLabel>

#include "Field.h"
#include "Format.h"

namespace dyno
{
	QBoolFieldWidget::QBoolFieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		FVar<bool>* f = TypeInfo::cast<FVar<bool>>(field);
		if (f == nullptr) {
			return;
		}

		//this->setStyleSheet("border:none");
		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedHeight(24);
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		QCheckBox* checkbox = new QCheckBox();
		checkbox->setFixedWidth(20);
		//checkbox->setFixedSize(40, 18);
		layout->addWidget(name, 0);
		layout->addStretch(1);
		layout->addWidget(checkbox, 0);

		connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(updateField(int)));

		checkbox->setChecked(f->getData());
	}

	void QBoolFieldWidget::updateField(int status)
	{
		FVar<bool>* f = TypeInfo::cast<FVar<bool>>(field());
		if (f == nullptr)
		{
			return;
		}

		if (status == Qt::Checked)
		{
			f->setValue(true);
			f->update();
		}
		else if (status == Qt::PartiallyChecked)
		{
			//m_pLabel->setText("PartiallyChecked");
		}
		else
		{
			f->setValue(false);
			f->update();
		}

		emit fieldChanged();
	}

}

