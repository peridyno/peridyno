#pragma once
#include <Wt/WContainerWidget.h>

#include <WParameterDataNode.h>

class WVector3FieldWidget : public Wt::WContainerWidget
{
public:
	WVector3FieldWidget(dyno::FBase*);
	//WVector3FieldWidget(std::string, dyno::Vec3f);
	~WVector3FieldWidget();

	static Wt::WContainerWidget* WVector3FieldWidgetConstructor(dyno::FBase* field)
	{
		return new WVector3FieldWidget(field);
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
	Wt::WDoubleSpinBox* mData1;
	Wt::WDoubleSpinBox* mData2;
	Wt::WDoubleSpinBox* mData3;

	dyno::Vec3f value;// active in "QVector3FieldWidget(QString name, Vec3f* v);"
	Wt::Signal<int> changeValue_;
};