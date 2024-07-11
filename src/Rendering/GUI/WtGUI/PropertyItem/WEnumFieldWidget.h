#pragma once
#include <Wt/WContainerWidget.h>
#include <Wt/WComboBox.h>

#include <WParameterDataNode.h>

#include "DeclareEnum.h"

class WEnumFieldWidget : public Wt::WContainerWidget
{
public:
	WEnumFieldWidget(dyno::FBase*);
	~WEnumFieldWidget();

	static Wt::WContainerWidget* WEnumFieldWidgetConstructor(dyno::FBase* field)
	{
		return new WEnumFieldWidget(field);
	};

	void setValue(dyno::FBase*);

	//Called when the widget is updated
	void updateField(int index);
	Wt::Signal<int>& changeValue()
	{
		return changeValue_;
	}

private:
	dyno::FBase* mfield;
	Wt::WHBoxLayout* layout;
	Wt::WComboBox* mData;

	std::map<int, int> mComboxIndexMap;
	Wt::Signal<int> changeValue_;
};