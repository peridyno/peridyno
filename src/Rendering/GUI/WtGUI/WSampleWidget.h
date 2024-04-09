#pragma once

#include <Wt/WContainerWidget.h>
#include <Wt/WSignal.h>

class Sample
{
public:
	virtual std::string name() const = 0;
	virtual std::string description() const { return name(); }
	virtual std::string thumbnail() const { return "logo.png"; }
	virtual std::string source() const = 0;
};

class WSampleWidget : public Wt::WContainerWidget
{
public:
	WSampleWidget();
	Wt::Signal<Sample*>& clicked();

private:
	Wt::Signal<Sample*> m_signal;

};