#pragma once
#include "Wt/WContainerWidget.h"
#include "Wt/WWidgetItem.h"
#include "Wt/WHBoxLayout.h"
#include "Wt/WLabel.h"
#include "Wt/WText.h"
#include <Wt/WDoubleSpinBox.h>
#include <Wt/WSignal.h>

#include <Field.h>
#include <WParameterDataNode.h>

class WRealFieldWidget : public Wt::WContainerWidget
{
public:

	WRealFieldWidget();
	WRealFieldWidget(dyno::FBase*);
	~WRealFieldWidget();

	static Wt::WContainerWidget* WRealFieldWidgetConstructor(dyno::FBase* field)
	{
		return new WRealFieldWidget(field);
	};

	void setValue(dyno::FBase* field);
	Wt::Signal<>& valueChanged() { return mSignal; };

	//Called when the field is updated
	void updateWidget();

	//Called when the widget is updated
	void updateField();

private:
	Wt::Signal<> mSignal;
	Wt::Signal<> fieldChanged;
	dyno::FBase* mfield;
	Wt::WHBoxLayout* layout;
	Wt::WDoubleSpinBox* mData;
	dyno::FVar<float>* mFloatField;
	dyno::FVar<double>* mDoubleField;
};