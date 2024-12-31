#pragma once
#include <Wt/WContainerWidget.h>
#include <Wt/WVBoxLayout.h>
#include <Wt/WText.h>
#include <Wt/WSignal.h>
#include <Wt/WTextArea.h>
#include <Wt/WLabel.h>
#include <Wt/WPushButton.h>

#include <cstdio>
#include "Log.h"
#include "WMainWindow.h"

class WLogMessage
{
public:
	WLogMessage();
	~WLogMessage();

	static void RecieveLogMessage(const dyno::Log::Message& m);

	void updateLog(const char* text);

	Wt::Signal<std::string>& updateText();

public:
	static WLogMessage* instance;

	Wt::WTextArea* edit;

	std::string message;

private:
	Wt::WText* mLog = nullptr;

	Wt::Signal<std::string> m_signal;

};


class WLogWidget : public Wt::WContainerWidget
{
public:
	WLogWidget(WMainWindow* parent);
	~WLogWidget();

	void showMessage(std::string s);

private:
	Wt::WTextArea* text;
	WMainWindow* mParent;
};