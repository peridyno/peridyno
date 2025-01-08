#include "QRealFieldWidget.h"

#include <QHBoxLayout>

#include "Field.h"
#include "QPiecewiseDoubleSpinBox.h"

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

		QToggleLabel* name = new QToggleLabel();
		name->setFixedHeight(24);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		slider = new QDoubleSlider;
		slider->setRange(field->getMin(), field->getMax());
		slider->setMinimumWidth(60);

		spinner = new QPiecewiseDoubleSpinBox();
		spinner->setRange(field->getMin(), field->getMax());
		spinner->setFixedWidth(100);

		std::string template_name = field->getTemplateName();
		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(field);

			spinner->setValue((double)f->getValue());
			slider->setValue((double)f->getValue());
		}
		else if (template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(field);

			spinner->setValue((double)f->getValue());
			slider->setValue(f->getValue());
		}

		layout->addWidget(name, 0);
		layout->addWidget(slider, 1);
		layout->addStretch();
		layout->addWidget(spinner, 2);
		layout->setSpacing(3);

		FormatFieldWidgetName(field->getObjectName());

		QObject::connect(slider, SIGNAL(valueChanged(double)), spinner, SLOT(ModifyValueAndUpdate(double)));
		QObject::connect(spinner, SIGNAL(valueChanged(double)), slider, SLOT(setValue(double)));
		QObject::connect(spinner, SIGNAL(valueChanged(double)), this, SLOT(updateField(double)));

		QObject::connect(name, SIGNAL(toggle(bool)), spinner, SLOT(toggleDecimals(bool)));

		QObject::connect(this, SIGNAL(fieldChanged()), this, SLOT(updateWidget()));
	}

	QRealFieldWidget::~QRealFieldWidget()
	{
	}

	void QRealFieldWidget::updateField(double value)
	{
		std::string template_name = field()->getTemplateName();

		double v = spinner->getRealValue();
		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(field());
			f->setValue((float)v);
			f->update();
		}
		else if (template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(field());
			f->setValue(v);
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

