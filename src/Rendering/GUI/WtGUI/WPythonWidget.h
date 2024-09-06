#pragma once

#include <Wt/WContainerWidget.h>
#include <Wt/WSignal.h>

namespace dyno
{
	class SceneGraph;
}

class WPythonWidget : public Wt::WContainerWidget
{
public:
	WPythonWidget();
	~WPythonWidget();

	void setText(const std::string& text);
	void execute(const std::string& src);


	Wt::Signal<std::shared_ptr<dyno::SceneGraph>>& updateSceneGraph() { return mSignal; }

private:
	Wt::WText* mCodeEditor;

	Wt::Signal<std::shared_ptr<dyno::SceneGraph>> mSignal;

};