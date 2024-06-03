#include "WBoolFieldWidget.h"

WBoolFieldWidget::WBoolFieldWidget(dyno::FBase* field)
{
	layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);

	setValue(field);
	mfield = field;

	checkbox->changed().connect(this, &WBoolFieldWidget::updateField);
}

WBoolFieldWidget::~WBoolFieldWidget()
{

}

void WBoolFieldWidget::setValue(dyno::FBase* field)
{
	dyno::FVar<bool>* f = TypeInfo::cast<dyno::FVar<bool>>(field);
	if (f == nullptr) {
		return;
	}
	checkbox->setChecked(f->getData());
}

void WBoolFieldWidget::updateField()
{
	dyno::FVar<bool>* f = TypeInfo::cast<dyno::FVar<bool>>(mfield);
	if (f == nullptr)
	{
		return;
	}
	f->setValue(false);
	f->update();
}