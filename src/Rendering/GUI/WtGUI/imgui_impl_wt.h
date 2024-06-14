#pragma once
#include "imgui.h"  

#include <Wt/WContainerWidget.h>


class ImGuiBackendWt
{
public:
	ImGuiBackendWt(Wt::WContainerWidget* parent);
	~ImGuiBackendWt();

	void NewFrame(int width, int height);

private:
	ImGuiContext* ctx;
};
