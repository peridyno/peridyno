#pragma once
#include <Wt/WContainerWidget.h>
#include <Wt/WLineEdit.h>
#include <Wt/WFileUpload.h>
#include <Wt/WPushButton.h>
#include <Wt/WString.h>
#include <Wt/WVBoxLayout.h>
#include <Wt/WMessageBox.h>

#include <WParameterDataNode.h>

#include <fstream>
#include <filesystem>

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
	Wt::Signal<int>& changeValue()
	{
		return changeValue_;
	}

private:
	dyno::FBase* mfield;
	Wt::WVBoxLayout* layout;
	Wt::WLineEdit* mfilename;
	//Wt::WPushButton* uploadButton;
	Wt::WFileUpload* upload;
	Wt::Signal<int> changeValue_;

	void uploadFile();
	void fileTooLarge();
	bool hasFile(std::string);
	std::string shortFilePath(std::string str);
};
