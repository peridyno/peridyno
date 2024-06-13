#include "imgui.h"
#include "imgui_impl_wt.h"
#include "imgui_internal.h"

ImGuiBackendWt::ImGuiBackendWt(Wt::WContainerWidget* parent)
{
	IMGUI_CHECKVERSION();
	ctx = ImGui::CreateContext();	
}

ImGuiBackendWt::~ImGuiBackendWt()
{
	//ImGui::DestroyContext(ctx);
}

void ImGuiBackendWt::NewFrame(int width, int height)
{
	ImGui::SetCurrentContext(ctx);
	auto& io = ctx->IO;
	io.DisplaySize.x = width;
	io.DisplaySize.y = height;
}
