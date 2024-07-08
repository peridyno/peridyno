#pragma once
#include "imgui.h"  

#include <Wt/WContainerWidget.h>


class ImGuiBackendWt
{
public:
	ImGuiBackendWt(Wt::WContainerWidget* parent);
	~ImGuiBackendWt();

	void NewFrame(int width, int height);
	void Render();

	bool handleMousePressed(const Wt::WMouseEvent& evt);
	bool handleMouseDrag(const Wt::WMouseEvent& evt);
	bool handleMouseReleased(const Wt::WMouseEvent& evt);

private:
	ImGuiContext* ctx;
};
