#pragma once
#include <Wt/WCheckBox.h>
#include <Wt/WContainerWidget.h>
#include <Field.h>
#include <Wt/WHBoxLayout.h>
#include <OBase.h>

class WStateFieldWidget : public Wt::WContainerWidget
{
public:
	WStateFieldWidget(dyno::FBase*);
	~WStateFieldWidget();

	void setValue(dyno::FBase*);

	void updateField();

	Wt::Signal<bool>& changeValue()
	{
		return changeValue_;
	}

private:
	dyno::FBase* mfield;
	Wt::WHBoxLayout* layout;
	Wt::WCheckBox* checkbox;
	Wt::Signal<bool> changeValue_;
};