#pragma once

#include <Wt/WContainerWidget.h>

namespace dyno
{
	class SceneGraph;
	class SceneGraphFactory;
	class Node;
};

class WNodeDataModel;
class WModuleDataModel;
class WParameterDataNode;
class WSimulationCanvas;
class WMainWindow : public Wt::WContainerWidget
{
public:
	WMainWindow();
	~WMainWindow();

	void setScene(std::shared_ptr<dyno::SceneGraph> scene);

private:
	void initMenu(Wt::WMenu*);
	void initLeftPanel(Wt::WContainerWidget*);

	void start();
	void stop();
	void step();
	void reset();

private:

	WSimulationCanvas* mSceneCanvas;

	// data models
	std::shared_ptr<WNodeDataModel>		mNodeDataModel;
	std::shared_ptr<WModuleDataModel>	mModuleDataModel;
	std::shared_ptr< WParameterDataNode> mParameterDataNode;

	bool	bRunFlag;
	bool	mReset;

	std::shared_ptr<dyno::SceneGraph>	mScene;
	std::shared_ptr<dyno::Node> mActiveNode;
};
