#include "WEnumFieldWidget.h"

WEnumFieldWidget::WEnumFieldWidget(dyno::FBase* field)
{
	layout = this->setLayout(std::make_unique<Wt::WHBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);

	setValue(field);
	mfield = field;
	mData->activated().connect(this, &WEnumFieldWidget::updateField);
}

WEnumFieldWidget::~WEnumFieldWidget() {}

void WEnumFieldWidget::setValue(dyno::FBase* field)
{
	auto f = TypeInfo::cast<dyno::FVar<dyno::PEnum>>(field);
	if (f == nullptr || f->getDataPtr() == nullptr) {
		return;
	}

	mData = layout->addWidget(std::make_unique<Wt::WComboBox>());

	auto& enums = f->getDataPtr()->enumMap();
	int num = 0;
	int curIndex = 0;
	for (auto e : enums)
	{
		mComboxIndexMap[num] = e.first;
		mData->addItem(e.second);

		if (e.first == f->getDataPtr()->currentKey())
		{
			curIndex = num;
		}

		num++;
	}

	mData->setCurrentIndex(curIndex);
}

void WEnumFieldWidget::updateField(int index)
{
	auto f = TypeInfo::cast<dyno::FVar<dyno::PEnum>>(mfield);
	if (f == nullptr || f->getDataPtr() == nullptr) {
		return;
	}

	auto& enums = f->getDataPtr()->enumMap();

	f->getDataPtr()->setCurrentKey(mComboxIndexMap[index]);

	f->update();
}