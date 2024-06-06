#pragma once
#include <Wt/WContainerWidget.h>
#include <Wt/WLineEdit.h>
#include <Wt/WFileUpload.h>
#include <Wt/WPushButton.h>
#include <Wt/WString.h>

#include <WParameterDataNode.h>

class WFileWidget : public Wt::WContainerWidget
{
public:
	WFileWidget(dyno::FBase*);
	~WFileWidget();

	static Wt::WContainerWidget* WFileWidgetConstructor(dyno::FBase* field)
	{
		return new WFileWidget(field);
	};

	void setValue(dyno::FBase*);

	//Called when the widget is updated
	void updateField();

	bool hasFile(std::string);

	std::string shortFilePath(std::string str);

private:
	dyno::FBase* mfield;
	Wt::WHBoxLayout* layout;
	Wt::WLineEdit* mfilename;
	//Wt::WFileUpload* fileUpload;
	Wt::WPushButton* button;
};
