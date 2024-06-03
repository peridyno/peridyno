#pragma once
#include <Wt/WContainerWidget.h>
#include <Wt/WSpinBox.h>

#include <WParameterDataNode.h>

template<typename T>
T castMaximum(float v) {
	T tMax = std::numeric_limits<T>::max();
	tMax = tMax < v ? tMax : (T)v;
	return tMax;
}

template<typename T>
T castMinimum(float v) {
	T tMin = std::numeric_limits<T>::min();
	tMin = tMin > v ? tMin : (T)v;
	return tMin;
}

class WIntegerFieldWidget : public Wt::WContainerWidget
{
public:
	WIntegerFieldWidget(dyno::FBase*);
	~WIntegerFieldWidget();

	static Wt::WContainerWidget* WIntegerFieldWidgetConstructor(dyno::FBase* field)
	{
		return new WIntegerFieldWidget(field);
	};

	void setValue(dyno::FBase*);

	//Called when the widget is updated
	void updateField();

private:
	dyno::FBase* mfield;
	Wt::WHBoxLayout* layout;
	Wt::WSpinBox* mData;
};

class WUIntegerFieldWidget : public Wt::WContainerWidget
{
public:
	WUIntegerFieldWidget(dyno::FBase*);
	~WUIntegerFieldWidget();

	static Wt::WContainerWidget* WUIntegerFieldWidgetConstructor(dyno::FBase* field)
	{
		return new WUIntegerFieldWidget(field);
	};

	void setValue(dyno::FBase*);

	//Called when the widget is updated
	void updateField();

private:
	dyno::FBase* mfield;
	Wt::WHBoxLayout* layout;
	Wt::WSpinBox* mData;
};