#pragma once

#include <Wt/WContainerWidget.h>

#include <SceneGraph.h>

#include "WSimulationCanvas.h"


class WSimulationControl : public Wt::WContainerWidget
{
public:
	WSimulationControl();
	~WSimulationControl();

	void setSceneCanvas(WSimulationCanvas* SceneCanvas) { mSceneCanvas = SceneCanvas; }
	void setSceneGraph(std::shared_ptr<dyno::SceneGraph> Scenes) { mScene = Scenes; }

public:
	void stop();

private:
	void start();
	void step();
	void reset();

private:
	WSimulationCanvas* mSceneCanvas = nullptr;
	std::shared_ptr<dyno::SceneGraph>	mScene = nullptr;

	bool	bRunFlag = false;
	bool	mReset;
};