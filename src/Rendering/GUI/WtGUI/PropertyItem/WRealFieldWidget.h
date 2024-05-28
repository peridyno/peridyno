#pragma once
#include "Wt/WContainerWidget.h"
#include "Wt/WWidgetItem.h"
#include "Wt/WHBoxLayout.h"
#include "Wt/WLabel.h"
#include "Wt/WText.h"

#include <FBase.h>

class WRealFieldWidget : public Wt::WContainerWidget
{
public:
	//DECLARE_FIELD_WIDGET
	WRealFieldWidget();
	WRealFieldWidget(dyno::FBase* field);
	~WRealFieldWidget();
};