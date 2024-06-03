#include "WVector3FieldWidget.h"

WVector3FieldWidget::WVector3FieldWidget(dyno::FBase* field)
	: Wt::WContainerWidget(), mData1(nullptr)
{
	layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);

	setValue(field);
	mfield = field;
	mData1->valueChanged().connect(this, &WVector3FieldWidget::updateField);
	mData2->valueChanged().connect(this, &WVector3FieldWidget::updateField);
	mData3->valueChanged().connect(this, &WVector3FieldWidget::updateField);
}

WVector3FieldWidget::~WVector3FieldWidget()
{
}

void WVector3FieldWidget::setValue(dyno::FBase* field)
{
	std::string template_name = field->getTemplateName();

	double v1 = 0;
	double v2 = 0;
	double v3 = 0;

	if (template_name == std::string(typeid(dyno::Vec3f).name()))
	{
		dyno::FVar<dyno::Vec3f>* f = TypeInfo::cast<dyno::FVar<dyno::Vec3f>>(field);
		auto v = f->getData();
		v1 = v[0];
		v2 = v[1];
		v3 = v[2];
	}
	else if (template_name == std::string(typeid(dyno::Vec3d).name()))
	{
		dyno::FVar<dyno::Vec3d>* f = TypeInfo::cast<dyno::FVar<dyno::Vec3d>>(field);
		auto v = f->getData();

		v1 = v[0];
		v2 = v[1];
		v3 = v[2];
	}

	mData1 = layout->addWidget(std::make_unique<Wt::WDoubleSpinBox>());
	mData2 = layout->addWidget(std::make_unique<Wt::WDoubleSpinBox>());
	mData3 = layout->addWidget(std::make_unique<Wt::WDoubleSpinBox>());

	mData1->setRange(field->getMin(), field->getMax());
	mData2->setRange(field->getMin(), field->getMax());
	mData3->setRange(field->getMin(), field->getMax());

	mData1->setSingleStep(0.01);
	mData2->setSingleStep(0.01);
	mData3->setSingleStep(0.01);

	mData1->setValue(v1);
	mData2->setValue(v2);
	mData3->setValue(v3);
}

void WVector3FieldWidget::updateField()
{
	double v1 = mData1->value();
	double v2 = mData2->value();
	double v3 = mData3->value();

	std::string template_name = mfield->getTemplateName();

	if (template_name == std::string(typeid(dyno::Vec3f).name()))
	{
		dyno::FVar<dyno::Vec3f>* f = TypeInfo::cast<dyno::FVar<dyno::Vec3f>>(mfield);
		f->setValue(dyno::Vec3f((float)v1, (float)v2, (float)v3));
	}
	else if (template_name == std::string(typeid(dyno::Vec3d).name()))
	{
		dyno::FVar<dyno::Vec3d>* f = TypeInfo::cast<dyno::FVar<dyno::Vec3d>>(mfield);
		f->setValue(dyno::Vec3d(v1, v2, v3));
	}
}