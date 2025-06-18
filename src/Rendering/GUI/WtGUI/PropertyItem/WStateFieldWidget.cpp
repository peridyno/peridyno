#include "WStateFieldWidget.h"

WStateFieldWidget::WStateFieldWidget(dyno::FBase* field)
{
	layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);

	setValue(field);
	mfield = field;

	checkbox->changed().connect(this, &WStateFieldWidget::updateField);
}

WStateFieldWidget::~WStateFieldWidget()
{
}

void WStateFieldWidget::setValue(dyno::FBase* field)
{
	checkbox = layout->addWidget(std::make_unique<Wt::WCheckBox>());
	field->parent()->findOutputField(field) ? checkbox->setChecked(true) : checkbox->setChecked(false);
}

void WStateFieldWidget::updateField()
{
	checkbox->isChecked() ? mfield->promoteOuput() : mfield->demoteOuput();
	
	changeValue_.emit(1);
}
