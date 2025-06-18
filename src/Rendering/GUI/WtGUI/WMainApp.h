#pragma once

#include <Wt/WApplication.h>

class WMainWindow;
class WMainApp : public Wt::WApplication
{
public:
	WMainApp(const Wt::WEnvironment& env);
	~WMainApp();

	static void mainLoop();

private:
	WMainWindow* window;
};