#include "WRealFieldWidget.h"



WRealFieldWidget::WRealFieldWidget()
	: Wt::WContainerWidget(), mData(nullptr), mFloatField(nullptr), mDoubleField(nullptr), layout(nullptr)
{
	auto layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);
	auto text = layout->addWidget(std::make_unique<Wt::WText>());
	text->setText("field->getObjectName()");
}

WRealFieldWidget::WRealFieldWidget(dyno::FBase* field)
	: Wt::WContainerWidget(), mData(nullptr), mFloatField(nullptr), mDoubleField(nullptr)
{
	layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);

	//mName = layout->addWidget(std::make_unique<Wt::WText>());
	//mName->setText(field->getObjectName());

	std::string template_name = field->getTemplateName();
	if (template_name == std::string(typeid(float).name()))
	{
		dyno::FVar<float>* f = TypeInfo::cast<dyno::FVar<float>>(field);
		//mFloatField = (double)f;
		mData = layout->addWidget(std::make_unique<Wt::WDoubleSpinBox>());
		mData->setValue((double)f->getValue());
	}
	else if (template_name == std::string(typeid(double).name()))
	{
		dyno::FVar<double>* f = TypeInfo::cast<dyno::FVar<double>>(field);
		mDoubleField = f;
		mData = layout->addWidget(std::make_unique<Wt::WDoubleSpinBox>());
		mData->setValue(mDoubleField->getValue());
	}
}

WRealFieldWidget::~WRealFieldWidget()
{
}

void WRealFieldWidget::setValue()
{
	if (mFloatField != nullptr || mDoubleField != nullptr)
	{
		mData->setValue(mFloatField != nullptr ? mFloatField->getValue()
			: mDoubleField->getValue());
	}
	else
	{
		Wt::log("warning") << "No Real Field";
	}
}