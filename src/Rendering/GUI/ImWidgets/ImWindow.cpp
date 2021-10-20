#include "ImWindow.h"

#include <imgui.h>
#include <imgui_impl_opengl3.h>

#include "imgui_extend.h"
#include "picture.h"
#include "imGuIZMO.quat/imGuIZMOquat.h"
//dyno
#include "Action.h"
#include "SceneGraph.h"

//ImWidgets
#include "ImWidget.h"

#include <Rendering.h>

using namespace dyno;

class WidgetQueue : public Action
{
private:
	void process(Node* node) override
	{
		if (!node->isVisible())
			return;

		for (auto iter : node->graphicsPipeline()->activeModules())
		{
			auto m = dynamic_cast<ImWidget*>(iter);
			if (m && m->isVisible())
			{
				//m->update();
				modules.push_back(m);
			}
		}
	}
public:
	std::vector<ImWidget*> modules;
};

void dyno::ImWindow::initialize(float scale)
{		
	// // TODO: Reorganize
	// mPics.emplace_back(std::make_shared<Picture>("../../data/icon/map.png"));
	// mPics.emplace_back(std::make_shared<Picture>("../../data/icon/box.png"));
	// mPics.emplace_back(std::make_shared<Picture>("../../data/icon/arrow-090-medium.png"));
	// mPics.emplace_back(std::make_shared<Picture>("../../data/icon/lock.png"));

	ImGui::initializeStyle(scale);
	ImGui::initColorVal();
}

void dyno::ImWindow::draw(RenderEngine* engine, SceneGraph* scene)
{

	RenderParams* rparams = engine->renderParams();

	float iBgGray[2] = { rparams->bgColor0[0], rparams->bgColor1[0] };
	RenderParams::Light iLight = rparams->light;
	
	// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
	{
		static float f = 0.0f;
		static int counter = 0;
		{// Top Left widget
			ImGui::SetNextWindowPos(ImVec2(0, 0));
			ImGui::Begin("Top Left widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

			if (ImGui::Button(ICON_FA_LIGHTBULB " Lighting")) {
				ImGui::OpenPopup("LightingMenu");
			}

			if (ImGui::BeginPopup("LightingMenu")) {
				ImGui::DragFloat2("BG color", iBgGray, 0.01f, 0.0f, 1.0f, "%.2f", 0);
				rparams->bgColor0 = glm::vec3(iBgGray[0]);
				rparams->bgColor1 = glm::vec3(iBgGray[1]);

				ImGui::Text("Ambient Light");

				ImGui::beginTitle("Ambient Light Scale");
				ImGui::DragFloat("", &iLight.ambientScale,0.01f, 0.0f, 1.0f, "%.2f", 0);
				ImGui::endTitle();
				ImGui::SameLine();
				ImGui::ColorEdit3("Ambient Light Color", (float*)&iLight.ambientColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float | ImGuiColorEditFlags_NoDragDrop | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoLabel);
				
				ImGui::Text("Main Light");
				ImGui::beginTitle("Main Light Scale");
				ImGui::DragFloat("", &iLight.mainLightScale, 0.01f, 0.0f, 5.0f, "%.2f", 0);
				ImGui::endTitle();
				ImGui::SameLine();
				ImGui::ColorEdit3("Main Light Color", (float*)&iLight.mainLightColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float | ImGuiColorEditFlags_NoDragDrop | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoLabel);

				// Light Direction
				ImGui::Text("Light Dir");
				
				glm::mat4 inverse_view = glm::transpose(rparams->view);// view R^-1 = R^T
				glm::vec3 tmpLightDir = glm::vec3(rparams->view * glm::vec4(iLight.mainLightDirection, 0));
				vec3 vL(-tmpLightDir[0], -tmpLightDir[1], -tmpLightDir[2]);

				ImGui::beginTitle("Light dir");
				ImGui::gizmo3D("", vL);
				ImGui::endTitle();
				
				ImGui::SameLine();
				ImGui::BeginGroup();
				ImGui::PushItemWidth(120*.5-2);
				ImGui::beginTitle("Light dir editor x");
				ImGui::DragFloat("", (float*)(&vL[0]), 0.01f, -1.0f, 1.0f, "x:%.2f", 0);ImGui::endTitle();
				ImGui::beginTitle("Light dir editor y");
				ImGui::DragFloat("", (float*)(&vL[1]), 0.01f, -1.0f, 1.0f, "y:%.2f", 0);ImGui::endTitle();
				ImGui::beginTitle("Light dir editor z");
				ImGui::DragFloat("", (float*)(&vL[2]), 0.01f, -1.0f, 1.0f, "z:%.2f", 0);ImGui::endTitle();
				ImGui::PopItemWidth();
				ImGui::EndGroup();

				tmpLightDir = glm::vec3(inverse_view * glm::vec4(vL[0], vL[1], vL[2], 0));
				iLight.mainLightDirection = glm::vec3(-tmpLightDir[0], -tmpLightDir[1], -tmpLightDir[2]);

				rparams->light = iLight;

				ImGui::EndPopup();
			}


			// Camera Select
			static int camera_current = 0;
			const char* camera_name[] = { ICON_FA_CAMERA " Orbit", ICON_FA_CAMERA " TrackBall" };
			static ImGuiComboFlags flags = ImGuiComboFlags_NoArrowButton;

			ImGui::SetNextItemWidth(ImGui::GetFrameHeight() * 4);
			ImGui::beginTitle("Camera");
			if (ImGui::BeginCombo("", camera_name[camera_current], flags))
			{
				for (int n = 0; n < IM_ARRAYSIZE(camera_name); n++)
				{
					const bool is_selected = (camera_current == n);
					if (ImGui::Selectable(camera_name[n], is_selected))
						camera_current = n;
					// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			ImGui::endTitle();

			// 					if(CameraType(camera_current) != mCameraType){
			// 						// TODO 
			// 						// setCameraType(CameraType(camera_current));
			// 					}
								//ImGui::ShowStyleEditor();
			ImGui::End();
		}

		{// Top Right widget

			ImGui::Begin("Top Right widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::toggleButton(ICON_FA_ARROWS_ALT " Lock", &(mDisenableCamera));
			// ImGui::toggleButton("Lock", &(mDisenableCamera));
			ImGui::SameLine();
			ImGui::toggleButton(ICON_FA_SQUARE " Ground", &(rparams->showGround));
			// ImGui::toggleButton("Ground", &(rparams->showGround));
			ImGui::SameLine();
			ImGui::toggleButton(ICON_FA_CUBE " Bounds", &(rparams->showSceneBounds));
			// ImGui::toggleButton("Bounds", &(rparams->showSceneBounds));
			ImGui::SameLine();
			ImGui::toggleButton(ICON_FA_LOCATION_ARROW " Axis Helper", &(rparams->showAxisHelper));
			// ImGui::toggleButton("Axis Helper", &(rparams->showAxisHelper));
			ImGui::SetWindowPos(ImVec2(rparams->viewport.w - ImGui::GetWindowSize().x, 0));

			ImGui::End();
		}

		// Bottom Right widget
		{
			std::string rEngineName = engine->name();
			ImGui::Begin("Bottom Left widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::Text("Rendered by %s: %.1f FPS", rEngineName.c_str(), ImGui::GetIO().Framerate);
			ImGui::SetWindowPos(ImVec2(rparams->viewport.w - ImGui::GetWindowSize().x, rparams->viewport.h - ImGui::GetWindowSize().y));
			ImGui::End();
		}
	}

	// Draw custom widgets
	// gather visual modules
	WidgetQueue imWidgetQueue;
	// enqueue render content
	if ((scene != 0) && (scene->getRootNode() != 0))
	{
		scene->getRootNode()->traverseTopDown(&imWidgetQueue);
	}

	for (auto widget : imWidgetQueue.modules)
	{
		widget->update();
		widget->paint();
	}
}

bool ImWindow::cameraLocked()
{
	return (ImGui::IsWindowFocused(ImGuiFocusedFlags_::ImGuiFocusedFlags_AnyWindow) || mDisenableCamera);
}