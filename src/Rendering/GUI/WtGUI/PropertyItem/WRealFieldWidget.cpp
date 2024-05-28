#include "WRealFieldWidget.h"

WRealFieldWidget::WRealFieldWidget()
	: Wt::WContainerWidget()
{
	auto layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);
	auto text = layout->addWidget(std::make_unique<Wt::WText>());
	text->setText("field->getObjectName()");
}

WRealFieldWidget::WRealFieldWidget(dyno::FBase* field)
	: Wt::WContainerWidget()
{
	auto layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);
	auto text = layout->addWidget(std::make_unique<Wt::WText>());
	text->setText(field->getObjectName());
}

WRealFieldWidget::~WRealFieldWidget()
{
}