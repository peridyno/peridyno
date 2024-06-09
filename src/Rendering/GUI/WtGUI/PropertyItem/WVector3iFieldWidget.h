#pragma once
#include <Wt/WContainerWidget.h>
#include <Wt/WSpinBox.h>

#include <WParameterDataNode.h>

class WVector3iFieldWidget : public Wt::WContainerWidget
{
public:
	WVector3iFieldWidget(dyno::FBase*);
	~WVector3iFieldWidget();

	static Wt::WContainerWidget* WVector3iFieldWidgetConstructor(dyno::FBase* field)
	{
		return new WVector3iFieldWidget(field);
	};

	void setValue(dyno::FBase*);

	//Called when the widget is updated
	void updateField();

private:
	dyno::FBase* mfield;
	Wt::WHBoxLayout* layout;
	Wt::WSpinBox* mData1;
	Wt::WSpinBox* mData2;
	Wt::WSpinBox* mData3;
};