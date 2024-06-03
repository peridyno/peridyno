#include "WVector3iFieldWidget.h"

WVector3iFieldWidget::WVector3iFieldWidget(dyno::FBase* field)
	: Wt::WContainerWidget()
{
	layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);

	setValue(field);
	mfield = field;

	mData1->valueChanged().connect(this, &WVector3iFieldWidget::updateField);
	mData2->valueChanged().connect(this, &WVector3iFieldWidget::updateField);
	mData3->valueChanged().connect(this, &WVector3iFieldWidget::updateField);
}

WVector3iFieldWidget::~WVector3iFieldWidget()
{
}

void WVector3iFieldWidget::setValue(dyno::FBase* field)
{
	std::string template_name = field->getTemplateName();

	int v1 = 0;
	int v2 = 0;
	int v3 = 0;

	if (template_name == std::string(typeid(dyno::Vec3i).name()))
	{
		dyno::FVar<dyno::Vec3i>* f = TypeInfo::cast<dyno::FVar<dyno::Vec3i>>(field);
		auto v = f->getData();
		v1 = v[0];
		v2 = v[1];
		v3 = v[2];
	}
	else if (template_name == std::string(typeid(dyno::Vec3u).name()))
	{
		dyno::FVar<dyno::Vec3u>* f = TypeInfo::cast<dyno::FVar<dyno::Vec3u>>(field);
		auto v = f->getData();

		v1 = v[0];
		v2 = v[1];
		v3 = v[2];
	}
	mData1 = layout->addWidget(std::make_unique<Wt::WSpinBox>());
	mData2 = layout->addWidget(std::make_unique<Wt::WSpinBox>());
	mData3 = layout->addWidget(std::make_unique<Wt::WSpinBox>());

	mData1->setValue(v1);
	mData2->setValue(v2);
	mData3->setValue(v3);
}

void WVector3iFieldWidget::updateField()
{
	int v1 = mData1->value();
	int v2 = mData2->value();
	int v3 = mData3->value();

	std::string template_name = mfield->getTemplateName();

	if (template_name == std::string(typeid(dyno::Vec3i).name()))
	{
		dyno::FVar<dyno::Vec3i>* f = TypeInfo::cast<dyno::FVar<dyno::Vec3i>>(mfield);
		f->setValue(dyno::Vec3i(v1, v2, v3));
	}
	else if (template_name == std::string(typeid(dyno::Vec3u).name()))
	{
		dyno::FVar<dyno::Vec3u>* f = TypeInfo::cast<dyno::FVar<dyno::Vec3u>>(mfield);
		f->setValue(dyno::Vec3u(v1, v2, v3));
	}
}