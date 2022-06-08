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

#include "OrbitCamera.h"
#include "TrackballCamera.h"

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
	ImGui::initializeStyle(scale);
	ImGui::initColorVal();
}

void ShowMenuFile(RenderEngine* engine, bool* mDisenableCamera)
{
	if (ImGui::BeginMenu("Camera")) {
		auto cam = engine->camera();
		int width = cam->viewportWidth();
		int height = cam->viewportHeight();
		if (ImGui::BeginMenu("use Camera.."))
		{
			if (ImGui::MenuItem("Orbit", "")) {
				auto oc = std::make_shared<OrbitCamera>();
				oc->setWidth(width);
				oc->setHeight(height);

				oc->setEyePos(Vec3f(1.5f, 1.0f, 1.5f));

				engine->setCamera(oc);
			}
			if (ImGui::MenuItem("Trackball", "")) {
				auto tc = std::make_shared<TrackballCamera>();

				tc->setWidth(width);
				tc->setHeight(height);

				tc->setEyePos(Vec3f(1.5f, 1.0f, 1.5f));

				engine->setCamera(tc);
			}
			ImGui::EndMenu();
		}

		ImGui::Separator();
		if (ImGui::MenuItem("Perspective", "")) {
			cam->setProjectionType(Camera::Perspective);
			
		}
		if (ImGui::MenuItem("Orthogonal", "")) {
			cam->setProjectionType(Camera::Orthogonal);
		}

		ImGui::Separator();
		Vec3f targetPos = cam->getTargetPos();
		Vec3f eyePos = cam->getEyePos();
		float Distance = (targetPos - eyePos).norm();

		if (ImGui::MenuItem("Top", "")) {
			cam->setEyePos(Vec3f(targetPos.x, Distance, targetPos.z));
			
		}
		
		if (ImGui::MenuItem("Bottom", "")) {
			cam->setEyePos(Vec3f(targetPos.x, -Distance, targetPos.z));
		}

		ImGui::Separator();
		if (ImGui::MenuItem("Left", "")) {
			cam->setEyePos(Vec3f(-Distance, targetPos.y, targetPos.z));
		}
		if (ImGui::MenuItem("Right", "")) {
			cam->setEyePos(Vec3f(Distance, targetPos.y, targetPos.z));
		}

		ImGui::Separator();
		if (ImGui::MenuItem("Front", "")) {
			cam->setEyePos(Vec3f(targetPos.x, targetPos.y, Distance));
		}

		if (ImGui::MenuItem("Back", "")) {
			cam->setEyePos(Vec3f(targetPos.x, targetPos.y, -Distance));
		}

		ImGui::EndMenu();
	}

	if (ImGui::BeginMenu("Lighting", "")) {

		RenderParams* rparams = engine->renderParams();
		float iBgGray[2] = { rparams->bgColor0[0], rparams->bgColor1[0] };
		RenderParams::Light iLight = rparams->light;

		ImGui::DragFloat2("BG color", iBgGray, 0.01f, 0.0f, 1.0f, "%.2f", 0);
		rparams->bgColor0 = glm::vec3(iBgGray[0]);
		rparams->bgColor1 = glm::vec3(iBgGray[1]);

		ImGui::Text("Ambient Light");

		ImGui::beginTitle("Ambient Light Scale");
		ImGui::DragFloat("", &iLight.ambientScale, 0.01f, 0.0f, 1.0f, "%.2f", 0);
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
		ImGui::PushItemWidth(120 * .5 - 2);
		ImGui::beginTitle("Light dir editor x");
		ImGui::DragFloat("", (float*)(&vL[0]), 0.01f, -1.0f, 1.0f, "x:%.2f", 0); ImGui::endTitle();
		ImGui::beginTitle("Light dir editor y");
		ImGui::DragFloat("", (float*)(&vL[1]), 0.01f, -1.0f, 1.0f, "y:%.2f", 0); ImGui::endTitle();
		ImGui::beginTitle("Light dir editor z");
		ImGui::DragFloat("", (float*)(&vL[2]), 0.01f, -1.0f, 1.0f, "z:%.2f", 0); ImGui::endTitle();
		ImGui::PopItemWidth();
		ImGui::EndGroup();

		tmpLightDir = glm::vec3(inverse_view * glm::vec4(vL[0], vL[1], vL[2], 0));
		iLight.mainLightDirection = glm::vec3(-tmpLightDir[0], -tmpLightDir[1], -tmpLightDir[2]);

		rparams->light = iLight;

	}
	if (ImGui::BeginMenu("Auxiliary", "")) {
			RenderParams* rparams = engine->renderParams();
			
			ImGui::Checkbox("Lock", mDisenableCamera);
			ImGui::Spacing();
			
			ImGui::Checkbox("Ground", &(rparams->showGround));
			ImGui::Spacing();

			ImGui::Checkbox("Bounds", &(rparams->showSceneBounds));
			ImGui::Spacing();

			ImGui::Checkbox("AxisHelper", &(rparams->showAxisHelper));
			ImGui::Spacing();

		ImGui::EndMenu();	
	}

}

void ShowExampleAppMainMenuBar(RenderEngine* engine, bool* mDisenableCamera)
{
	if (ImGui::BeginMainMenuBar())
	{
		ShowMenuFile(engine, mDisenableCamera);
		ImGui::EndMainMenuBar();
	}
}


void dyno::ImWindow::draw(RenderEngine* engine, SceneGraph* scene)
{

	RenderParams* rparams = engine->renderParams();

	float iBgGray[2] = { rparams->bgColor0[0], rparams->bgColor1[0] };
	RenderParams::Light iLight = rparams->light;
	
	
	// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
	{

		ShowExampleAppMainMenuBar(engine, &mDisenableCamera);
		// Bottom Right widget
		{
			std::string rEngineName = engine->name();
			ImGui::Begin("Bottom Left widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::Text("Rendered by %s: %.1f FPS", rEngineName.c_str(), ImGui::GetIO().Framerate);
			ImGui::SetWindowPos(ImVec2(rparams->viewport.w - ImGui::GetWindowSize().x, rparams->viewport.h - ImGui::GetWindowSize().y));
			ImGui::End();
		}

		//----------------------------------------------------------------------------------------------------------------
		IM_ASSERT(ImGui::GetCurrentContext() != NULL && "Missing dear imgui context. Refer to examples app!");
		/*
		ImGui::Spacing();
		
		if (ImGui::CollapsingHeader("Style"))
		{
			ImGuiIO& io = ImGui::GetIO();

			if (ImGui::TreeNode("Style"))
			{
				ImGui::ShowStyle();
				ImGui::TreePop();
				ImGui::Separator();
			}
		}
		*/
	}
	
	// Draw custom widgets
	// gather visual modules
	WidgetQueue imWidgetQueue;
	// enqueue render content
	if (!scene->isEmpty())
	{
		scene->traverseForward(&imWidgetQueue);
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