#pragma once
#include "Wt/WContainerWidget.h"
#include "Wt/WWidgetItem.h"
#include "Wt/WHBoxLayout.h"
#include "Wt/WLabel.h"
#include "Wt/WText.h"
#include <Wt/WDoubleSpinBox.h>

#include <FBase.h>
#include <Field.h>
#include <WParameterDataNode.h>

class WRealFieldWidget : public Wt::WContainerWidget
{
public:

	WRealFieldWidget();
	WRealFieldWidget(dyno::FBase* field);
	~WRealFieldWidget();

	static Wt::WContainerWidget* WRealFieldWidgetConstructor(dyno::FBase* field)
	{
		return new WRealFieldWidget(field);
	};

	void setValue();
	Wt::Signal<double>& valueChanged();

private:

	Wt::WHBoxLayout* layout;
	Wt::WDoubleSpinBox* mData = nullptr;
	dyno::FVar<float>* mFloatField = nullptr;
	dyno::FVar<double>* mDoubleField = nullptr;
};