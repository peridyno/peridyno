#pragma once
#include <Wt/WContainerWidget.h>
#include <Wt/WVBoxLayout.h>
#include <Wt/WPushButton.h>
#include <SceneGraphFactory.h>
#include <SceneLoaderFactory.h>


namespace dyno
{
	class SceneLoaderFactory;
	class SceneGraphFactory;
}

class WSaveWidget : public Wt::WContainerWidget
{
public:
	WSaveWidget();
	~WSaveWidget();

private:
	void save();
};