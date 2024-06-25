#include "WIntegerFieldWidget.h"

WIntegerFieldWidget::WIntegerFieldWidget(dyno::FBase* field)
{
	layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);

	setValue(field);
	mfield = field;
	mData->valueChanged().connect(this, &WIntegerFieldWidget::updateField);
}

WIntegerFieldWidget::~WIntegerFieldWidget() {}

void WIntegerFieldWidget::setValue(dyno::FBase* field)
{
	dyno::FVar<int>* f = TypeInfo::cast<dyno::FVar<int>>(field);
	if (f == nullptr)
		return;

	mData = layout->addWidget(std::make_unique<Wt::WSpinBox>());
	mData->setRange(castMinimum<int>(field->getMin()), castMaximum<int>(field->getMax()));
	mData->setSingleStep(1);
	mData->setValue(f->getData());
}

void WIntegerFieldWidget::updateField()
{
	int v = mData->value();
	dyno::FVar<int>* f = TypeInfo::cast<dyno::FVar<int>>(mfield);
	if (f == nullptr)
		return;

	f->setValue(v);
	changeValue_.emit(1);
}

WUIntegerFieldWidget::WUIntegerFieldWidget(dyno::FBase* field)
{
	layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);

	setValue(field);
	mfield = field;
	mData->valueChanged().connect(this, &WUIntegerFieldWidget::updateField);
}

WUIntegerFieldWidget::~WUIntegerFieldWidget() {}

void WUIntegerFieldWidget::setValue(dyno::FBase* field)
{
	dyno::FVar<dyno::uint>* f = TypeInfo::cast<dyno::FVar<dyno::uint>>(field);
	if (f == nullptr)
		return;

	mData = layout->addWidget(std::make_unique<Wt::WSpinBox>());
	mData->setRange(castMinimum<dyno::uint>(field->getMin()), castMaximum<dyno::uint>(field->getMax()));
	mData->setSingleStep(1);
	mData->setValue(f->getData());
}

void WUIntegerFieldWidget::updateField()
{
	int v = mData->value();
	dyno::FVar<dyno::uint>* f = TypeInfo::cast<dyno::FVar<dyno::uint>>(mfield);
	if (f == nullptr)
		return;

	f->setValue(v);
	changeValue_.emit(1);
}