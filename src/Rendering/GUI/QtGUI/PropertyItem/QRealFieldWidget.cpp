#include "QRealFieldWidget.h"

#include <QHBoxLayout>

#include "Field.h"

namespace dyno
{
	IMPL_FIELD_WIDGET(float, QRealFieldWidget)

	QRealFieldWidget::QRealFieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		//this->setStyleSheet("border:none");
		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedHeight(24);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		slider = new QDoubleSlider;
		slider->setRange(field->getMin(), field->getMax());
		slider->setMinimumWidth(60);

		QDoubleSpinner* spinner = new QDoubleSpinner;
		spinner->setRange(field->getMin(), field->getMax());
		spinner->setFixedWidth(100);
		spinner->setDecimals(3);

		layout->addWidget(name, 0);
		layout->addWidget(slider, 1);
		layout->addStretch();
		layout->addWidget(spinner, 2);
		layout->setSpacing(3);

		std::string template_name = field->getTemplateName();
		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(field);
			slider->setValue((double)f->getValue());
			spinner->setValue((double)f->getValue());
		}
		else if(template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(field);
			slider->setValue(f->getValue());
			spinner->setValue(f->getValue());
		}

		FormatFieldWidgetName(field->getObjectName());

		QObject::connect(slider, SIGNAL(valueChanged(double)), spinner, SLOT(setValue(double)));
		QObject::connect(spinner, SIGNAL(valueChanged(double)), slider, SLOT(setValue(double)));
		QObject::connect(spinner, SIGNAL(valueChanged(double)), this, SLOT(updateField(double)));
	}

	QRealFieldWidget::~QRealFieldWidget()
	{
	}

	void QRealFieldWidget::updateField(double value)
	{
		std::string template_name = field()->getTemplateName();

		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(field());
			f->setValue((float)value);
			f->update();
		}
		else if (template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(field());
			f->setValue(value);
			f->update();
		}

		emit fieldChanged();
	}

	void QRealFieldWidget::updateWidget()
	{
		std::string template_name = field()->getTemplateName();
		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(field());

			slider->blockSignals(true);
			slider->setValue((double)f->getValue());
			slider->blockSignals(false);
		}
		else if (template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(field());

			slider->blockSignals(true);
			slider->setValue(f->getValue());
			slider->blockSignals(false);
		}
	}
}

