#include "ImWidget.h"

namespace dyno
{
	ImWidget::ImWidget()
		: VisualModule()
	{

	}

	ImWidget::~ImWidget()
	{

	}

	void ImWidget::updateGraphicsContext()
	{
		this->update();
	}
}