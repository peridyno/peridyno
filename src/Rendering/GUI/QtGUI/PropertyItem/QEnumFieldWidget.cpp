#include "QEnumFieldWidget.h"

#include <QHBoxLayout>
#include <QComboBox>

#include "DeclareEnum.h"

namespace dyno
{
	IMPL_FIELD_WIDGET(PEnum, QEnumFieldWidget)

		QEnumFieldWidget::QEnumFieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		auto f = TypeInfo::cast<FVar<PEnum>>(field);
		if (f == nullptr || f->getDataPtr() == nullptr) {
			return;
		}

		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		QLabel* name = new QLabel();
		name->setFixedHeight(24);
		name->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		layout->addWidget(name, 0);

		QComboBox* combox = new QComboBox;
		combox->setMaximumWidth(256);

		auto& enums = f->getDataPtr()->enumMap();
		int num = 0;
		int curIndex = 0;
		for (auto e : enums)
		{
			mComboxIndexMap[num] = e.first;
			combox->addItem(QString::fromStdString(e.second));

			if (e.first == f->getDataPtr()->currentKey()) {
				curIndex = num;
			}

			num++;
		}

		combox->setCurrentIndex(curIndex);

		layout->addWidget(combox, 1);

		this->setLayout(layout);

		connect(combox, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &QEnumFieldWidget::updateField);
	}

	void QEnumFieldWidget::updateField(int index)
	{
		auto f = TypeInfo::cast<FVar<PEnum>>(field());
		if (f == nullptr || f->getDataPtr() == nullptr) {
			return;
		}

		auto& enums = f->getDataPtr()->enumMap();

		f->getDataPtr()->setCurrentKey(mComboxIndexMap[index]);
		//To notify the field is updated
		f->update();
	}
}