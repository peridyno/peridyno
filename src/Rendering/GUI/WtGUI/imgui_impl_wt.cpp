#include "imgui_impl_wt.h"

#include "imgui_impl_opengl3.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <Wt/WEvent.h>

ImGuiBackendWt::ImGuiBackendWt(Wt::WContainerWidget* parent)
{
	IMGUI_CHECKVERSION();
	ctx = ImGui::CreateContext();
	ImGui::SetCurrentContext(ctx);

	const char* glsl_version = "#version 130";
	ImGui_ImplOpenGL3_Init(glsl_version);
}

ImGuiBackendWt::~ImGuiBackendWt()
{
	ImGui::SetCurrentContext(ctx);
	ImGui_ImplOpenGL3_Shutdown();
	ImGui::DestroyContext(ctx);
}

void ImGuiBackendWt::NewFrame(int width, int height)
{
	ImGui::SetCurrentContext(ctx);
	auto& io = ImGui::GetIO();
	io.DisplaySize.x = width;
	io.DisplaySize.y = height;

	ImGui_ImplOpenGL3_NewFrame();
	ImGui::NewFrame();

	//while (ctx->InputEventsQueue.Size > 0)
	//{
	//	ImGui::EndFrame();
	//	ImGui::NewFrame();
	//}
}

void ImGuiBackendWt::Render()
{
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool ImGuiBackendWt::handleMousePressed(const Wt::WMouseEvent& evt)
{
	auto& io = ctx->IO;
	int button = -1;
	if (evt.button() == Wt::MouseButton::Left) button = 0;
	if (evt.button() == Wt::MouseButton::Right) button = 1;
	if (evt.button() == Wt::MouseButton::Middle) button = 2;
	//io.AddMousePosEvent(evt.widget().x, evt.widget().y);
	io.MousePos = ImVec2(evt.widget().x, evt.widget().y);
	io.AddMouseButtonEvent(button, true);
	return io.WantCaptureMouse;
}

bool ImGuiBackendWt::handleMouseDrag(const Wt::WMouseEvent& evt)
{
	auto& io = ctx->IO;
	io.AddMousePosEvent(evt.widget().x, evt.widget().y);
	return io.WantCaptureMouse;
}

bool ImGuiBackendWt::handleMouseReleased(const Wt::WMouseEvent& evt)
{
	auto& io = ctx->IO;
	int button = -1;
	if (evt.button() == Wt::MouseButton::Left) button = 0;
	if (evt.button() == Wt::MouseButton::Right) button = 1;
	if (evt.button() == Wt::MouseButton::Middle) button = 2;
	//io.AddMousePosEvent(evt.widget().x, evt.widget().y);
	io.MousePos = ImVec2(evt.widget().x, evt.widget().y);
	io.AddMouseButtonEvent(button, false);
	return io.WantCaptureMouse;
}

