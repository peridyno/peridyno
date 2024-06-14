#include "WColorWidget.h"

WColorWidget::WColorWidget(dyno::FBase* field)
{
	layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);

	setValue(field);
	mfield = field;
	mData->colorInput().connect(this, &WColorWidget::updateField);
}

WColorWidget::~WColorWidget() {}

void WColorWidget::setValue(dyno::FBase* field)
{
	mData = layout->addWidget(std::make_unique<Wt::WColorPicker>());

	std::string template_name = field->getTemplateName();
	int R = 0;
	int G = 0;
	int B = 0;

	if (template_name == std::string(typeid(dyno::Color).name()))
	{
		dyno::FVar<dyno::Color>* f = TypeInfo::cast<dyno::FVar<dyno::Color>>(field);
		auto v = f->getData();

		int r = int(v.r * 255) % 255;
		int g = int(v.g * 255) % 255;
		int b = int(v.b * 255) % 255;

		auto color = Wt::WColor(r, g, b);
		mData->setColor(color);
	}
}

void WColorWidget::updateField()
{
	std::string template_name = mfield->getTemplateName();

	if (template_name == std::string(typeid(dyno::Color).name()))
	{
		auto color = mData->color();
		int v1 = color.red();
		int v2 = color.green();
		int v3 = color.blue();

		dyno::FVar<dyno::Color>* f = TypeInfo::cast<dyno::FVar<dyno::Color>>(mfield);
		float r = float(v1) / 255;
		float g = float(v2) / 255;
		float b = float(v3) / 255;

		f->setValue(dyno::Color(r, g, b));
	}
}