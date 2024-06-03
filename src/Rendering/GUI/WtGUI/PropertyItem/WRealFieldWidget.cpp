#include "WRealFieldWidget.h"

WRealFieldWidget::WRealFieldWidget(dyno::FBase* field)
	: Wt::WContainerWidget(), mData(nullptr), mFloatField(nullptr), mDoubleField(nullptr)
{
	layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);

	setValue(field);

	mfield = field;
	mData->valueChanged().connect(this, &WRealFieldWidget::updateField);
}

WRealFieldWidget::~WRealFieldWidget()
{
}

void WRealFieldWidget::setValue(dyno::FBase* field)
{
	std::string template_name = field->getTemplateName();
	if (template_name == std::string(typeid(float).name()))
	{
		dyno::FVar<float>* f = TypeInfo::cast<dyno::FVar<float>>(field);
		mFloatField = f;
		mData = layout->addWidget(std::make_unique<Wt::WDoubleSpinBox>());
		mData->setRange(field->getMin(), field->getMax());
		mData->setSingleStep(0.0001);
		mData->setDecimals(4);
		mData->setValue((double)f->getValue());
	}
	else if (template_name == std::string(typeid(double).name()))
	{
		dyno::FVar<double>* f = TypeInfo::cast<dyno::FVar<double>>(field);
		mDoubleField = f;
		mData = layout->addWidget(std::make_unique<Wt::WDoubleSpinBox>());
		mData->setRange(field->getMin(), field->getMax());
		mData->setSingleStep(0.0001);
		mData->setDecimals(4);
		mData->setValue(mDoubleField->getValue());
	}
}

void WRealFieldWidget::updateField()
{
	std::string template_name = mfield->getTemplateName();
	double v = mData->value();
	if (template_name == std::string(typeid(float).name()))
	{
		dyno::FVar<float>* f = TypeInfo::cast<dyno::FVar<float>>(mfield);
		f->setValue((float)v);
		f->update();
	}
	else if (template_name == std::string(typeid(double).name()))
	{
		dyno::FVar<double>* f = TypeInfo::cast<dyno::FVar<double>>(mfield);
		f->setValue(v);
		f->update();
	};
}