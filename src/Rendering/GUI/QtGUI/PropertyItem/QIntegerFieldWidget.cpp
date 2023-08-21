#include "QIntegerFieldWidget.h"

#include <QGridLayout>

#include "Field.h"

namespace dyno
{
	IMPL_FIELD_WIDGET(int, QIntegerFieldWidget)
	IMPL_FIELD_WIDGET(uint, QUIntegerFieldWidget)

	QIntegerFieldWidget::QIntegerFieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		FVar<int>* f = TypeInfo::cast<FVar<int>>(field);
		if (f == nullptr) {
			return;
		}

		//this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		QString str = FormatFieldWidgetName(field->getObjectName());
		name->setFixedSize(100, 18);
		QFontMetrics fontMetrics(name->font());
		QString elide = fontMetrics.elidedText(str, Qt::ElideRight, 100);
		name->setText(elide);
		//Set label tips
		name->setToolTip(str);

		QSpinBox* spinner = new QSpinBox;
		spinner->setValue(f->getData());

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner, 0, 1, Qt::AlignRight);


		this->connect(spinner, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));
	}

	void QIntegerFieldWidget::changeValue(int value)
	{
		FVar<int>* f = TypeInfo::cast<FVar<int>>(field());
		if (f == nullptr)
			return;

		f->setValue(value);
	}

	QUIntegerFieldWidget::QUIntegerFieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		FVar<uint>* f = TypeInfo::cast<FVar<uint>>(field);
		if (f == nullptr)
		{
			return;
		}

		//this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedSize(100, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		QSpinBox* spinner = new QSpinBox;
		spinner->setMaximum(1000000);
		spinner->setFixedWidth(100);
		spinner->setValue(f->getData());

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner, 0, 1, Qt::AlignRight);
		layout->setSpacing(3);

		this->connect(spinner, SIGNAL(valueChanged(int)), this, SLOT(changeValue(int)));
	}

	void QUIntegerFieldWidget::changeValue(int value)
	{
		FVar<uint>* f = TypeInfo::cast<FVar<uint>>(field());
		if (f == nullptr)
			return;

		f->setValue(value);
	}
}

