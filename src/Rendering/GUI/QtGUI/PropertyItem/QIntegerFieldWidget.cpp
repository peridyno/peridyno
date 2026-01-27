#include "QIntegerFieldWidget.h"

#include <QGridLayout>

#include "Field.h"
#include "QPiecewiseSpinBox.h"

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

		QPiecewiseSpinBox* spinner = new QPiecewiseSpinBox;
		spinner->setRange(castMinimum<int>(field->getMin()), castMaximum<int>(field->getMax()));

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner, 0, 1, Qt::AlignRight);


		this->connect(spinner, SIGNAL(valueChanged(int)), this, SLOT(updateField(int)));

	}

	void QIntegerFieldWidget::updateField(int value)
	{

		FVar<int>* f = TypeInfo::cast<FVar<int>>(field());
		if (f == nullptr)
			return;

		f->setValue(value,false);
		emit fieldChanged();

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
		name->setFixedHeight(18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		QPiecewiseSpinBox* spinner = new QPiecewiseSpinBox;
		spinner->setFixedWidth(100);
		spinner->setRange(castMinimum<uint>(field->getMin()), castMaximum<int>(field->getMax()));
		spinner->setValue(f->getValue());

		layout->addWidget(name, 0, 0);
		layout->addWidget(spinner, 0, 1, Qt::AlignRight);
		layout->setSpacing(3);

		this->connect(spinner, SIGNAL(valueChanged(int)), this, SLOT(updateField(int)));
	}

	void QUIntegerFieldWidget::updateField(int value)
	{
		FVar<uint>* f = TypeInfo::cast<FVar<uint>>(field());
		if (f == nullptr)
			return;

		f->setValue(value,false);
		emit fieldChanged();
	}
}

