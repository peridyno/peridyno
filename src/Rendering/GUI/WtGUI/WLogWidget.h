#pragma once
#include <Wt/WContainerWidget.h>
#include <Wt/WVBoxLayout.h>
#include <Wt/WText.h>

class WLogWidget : public Wt::WContainerWidget
{
public:
	WLogWidget();
	~WLogWidget();

private:
	Wt::WText* mCodeEditor;
};