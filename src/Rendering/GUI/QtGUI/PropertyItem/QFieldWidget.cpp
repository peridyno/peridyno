#include "QFieldWidget.h"

#include "Module.h"
#include "Node.h"
#include "Field.h"
#include "SceneGraphFactory.h"

#include <QVBoxLayout>

#include <memory>


namespace dyno
{
	QFieldWidget::QFieldWidget(FBase* field)
	{
		mField = field;

		if (mField != nullptr)
		{
			callback = std::make_shared<FCallBackFunc>(std::bind(&QFieldWidget::syncValueFromField, this));
			mField->attach(callback);
		}
	}

	QFieldWidget::~QFieldWidget()
	{
		this->clearCallBackFunc();
	}

	void QFieldWidget::clearCallBackFunc()
	{
		if (mField != nullptr) {
			mField->detach(callback);
		}
	}

	void QFieldWidget::syncValueFromField()
	{
		auto node = dynamic_cast<Node*>(mField->parent());
		if (node != nullptr) {
			node->updateGraphicsContext();
		}

		emit fieldChanged();
	}



}

