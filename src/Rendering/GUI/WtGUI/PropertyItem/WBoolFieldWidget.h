#pragma once
#include <Wt/WContainerWidget.h>
#include <Wt/WCheckBox.h>

#include <Field.h>
#include <WParameterDataNode.h>

class WBoolFieldWidget : public Wt::WContainerWidget
{
public:
	WBoolFieldWidget(dyno::FBase*);
	~WBoolFieldWidget();

	static Wt::WContainerWidget* WBoolFieldWidgetConstructor(dyno::FBase* field)
	{
		return new WBoolFieldWidget(field);
	};


	void setValue(dyno::FBase*);

	//Called when the widget is updated
	void updateField();

	Wt::Signal<int>& changeValue()
	{
		return changeValue_;
	}

private:
	dyno::FBase* mfield;
	Wt::WHBoxLayout* layout;
	Wt::WCheckBox* checkbox;
	Wt::Signal<int> changeValue_;
};