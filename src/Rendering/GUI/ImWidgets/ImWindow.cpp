#include "ImWindow.h"

#include <cmath>
#include <imgui.h>
#include <imgui_impl_opengl3.h>

#include <glm/gtx/string_cast.hpp>

#include "imgui_extend.h"
#include "picture.h"
#include "imGuIZMO.quat/imGuIZMOquat.h"
//dyno
#include "Action.h"
#include "SceneGraphFactory.h"

//ImWidgets
#include "ImWidget.h"

#include <RenderEngine.h>
#include <RenderWindow.h>

#include "OrbitCamera.h"
#include "TrackballCamera.h"

#include "imgui.h"

#include "ImGuizmo.h"

#include <Node/ParametricModel.h>

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
			auto m = dynamic_cast<ImWidget*>(iter.get());
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

void ShowMenuFile(RenderWindow* app, SceneGraph* scene, bool* mDisenableCamera)
{

}


void ImWindow::draw(RenderWindow* app)
{
	auto engine = app->getRenderEngine();
	auto scene  = SceneGraphFactory::instance()->active();
	auto camera = app->getCamera();

	auto& rparams = app->getRenderParams();

	ImGuiIO& io = ImGui::GetIO();


	// Initialize ImGuizmo frame
	ImGuizmo::BeginFrame();
	ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
	
	// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
	{
		if (ImGui::BeginMainMenuBar())
		{

			if (ImGui::BeginMenu("Camera")) {
				if (ImGui::BeginMenu("Use Camera.."))
				{
					if (ImGui::MenuItem("Orbit", "")) {
						auto oc = std::make_shared<OrbitCamera>();
						oc->setWidth(camera->viewportWidth());
						oc->setHeight(camera->viewportHeight());
						oc->setEyePos(Vec3f(1.5f, 1.0f, 1.5f));
						app->setCamera(oc);
					}
					if (ImGui::MenuItem("Trackball", "")) {
						auto tc = std::make_shared<TrackballCamera>();
						tc->setWidth(camera->viewportWidth());
						tc->setHeight(camera->viewportHeight());
						tc->setEyePos(Vec3f(1.5f, 1.0f, 1.5f));
						app->setCamera(tc);
					}
					ImGui::EndMenu();
				}

				ImGui::Separator();
				if (ImGui::MenuItem("Perspective", "")) {
					camera->setProjectionType(Camera::Perspective);

				}
				if (ImGui::MenuItem("Orthogonal", "")) {
					camera->setProjectionType(Camera::Orthogonal);
				}

				ImGui::Separator();
				Vec3f tarPos = camera->getTargetPos();
				Vec3f eyePos = camera->getEyePos();
				float distance = (tarPos - eyePos).norm();

				if (ImGui::MenuItem("Top", "")) {
					camera->setEyePos(Vec3f(tarPos.x, tarPos.y + distance, tarPos.z));
				}

				if (ImGui::MenuItem("Bottom", "")) {
					camera->setEyePos(Vec3f(tarPos.x, tarPos.y - distance, tarPos.z));
				}

				ImGui::Separator();
				if (ImGui::MenuItem("Left", "")) {
					camera->setEyePos(Vec3f(tarPos.x - distance, tarPos.y, tarPos.z));
				}

				if (ImGui::MenuItem("Right", "")) {
					camera->setEyePos(Vec3f(tarPos.x + distance, tarPos.y, tarPos.z));
				}

				ImGui::Separator();
				if (ImGui::MenuItem("Front", "")) {
					camera->setEyePos(Vec3f(tarPos.x, tarPos.y, tarPos.z + distance));
				}

				if (ImGui::MenuItem("Back", "")) {
					camera->setEyePos(Vec3f(tarPos.x, tarPos.y, tarPos.z - distance));
				}

				ImGui::Separator();
				if (ImGui::Button("Auto Focus"))
				{
					if (scene) {
						auto box = scene->boundingBox();

						float len = box.maxLength();
						Vec3f center = 0.5f * (box.upper + box.lower);

						Vec3f eyePos = camera->getEyePos();
						Vec3f tarPos = camera->getTargetPos();
						Vec3f dir = eyePos - tarPos;
						dir.normalize();

						camera->setEyePos(center + len * dir);
						camera->setTargetPos(center);

						float unit = std::floor(std::log(len));
						camera->setUnitScale(std::pow(10.0f, (float)unit));
					}

				}

				float distanceUnit = camera->unitScale();
				if (ImGui::DragFloat("DistanceUnit", &distanceUnit, 0.01f, 10.0f))
					camera->setUnitScale(distanceUnit);

				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("Lighting", "")) {

				float iBgGray[2] = { rparams.bgColor0[0], rparams.bgColor1[0] };
				RenderParams::Light iLight = rparams.light;

				ImGui::DragFloat2("BG color", iBgGray, 0.01f, 0.0f, 1.0f, "%.2f", 0);
				rparams.bgColor0 = glm::vec3(iBgGray[0]);
				rparams.bgColor1 = glm::vec3(iBgGray[1]);

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
				ImGui::Text("Main Light Direction");

				glm::mat4 view = camera->getViewMat();
				glm::mat4 inverse_view = glm::transpose(view);// view R^-1 = R^T
				glm::vec3 tmpLightDir = glm::vec3(view * glm::vec4(iLight.mainLightDirection, 0));
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

				// Light Shadow
				bool shadow = iLight.mainLightShadow != 0;
				if (ImGui::Checkbox("Main Light Shadow", &shadow))
					iLight.mainLightShadow = shadow ? 1.f : 0.f;

				rparams.light = iLight;

				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("Auxiliary", "")) {

				ImGui::Checkbox("Lock Camera", &mDisenableCamera);
				ImGui::Spacing();

				ImGui::Checkbox("Show View Manipulator", &mViewManipulator);
				ImGui::Spacing();

				ImGui::Checkbox("Show Background", &(rparams.showGround));
				ImGui::Spacing();

				ImGui::Separator();

				if (scene) {
					bool canPrintNodeInfo = scene->isNodeInfoPrintable();
					if (ImGui::Checkbox("Print Node Info", &canPrintNodeInfo))
						scene->printNodeInfo(canPrintNodeInfo);

					bool canPrintModuleInfo = scene->isModuleInfoPrintable();
					if (ImGui::Checkbox("Print Module Info", &canPrintModuleInfo))
						scene->printModuleInfo(canPrintModuleInfo);

					ImGui::Separator();

					ImGui::Checkbox("Show Bounding Box", &(rparams.showSceneBounds));
					ImGui::Spacing();

					Vec3f lowerBound = scene->getLowerBound();
					float lo[3] = { lowerBound[0], lowerBound[1], lowerBound[2] };
					ImGui::InputFloat3("Lower Bound", lo);
					scene->setLowerBound(Vec3f(lo[0], lo[1], lo[2]));

					Vec3f upperBound = scene->getUpperBound();
					float up[3] = { upperBound[0], upperBound[1], upperBound[2] };
					ImGui::InputFloat3("Upper Bound", up);
					scene->setUpperBound(Vec3f(up[0], up[1], up[2]));
				}


				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("Edit", "")) {

				if (ImGui::RadioButton("Translate", mEditMode == 0))
					mEditMode = 0;

				if (ImGui::RadioButton("Scale", mEditMode == 1))
					mEditMode = 1;

				if (ImGui::RadioButton("Rotate", mEditMode == 2))
					mEditMode = 2;				

				ImGui::EndMenu();
			}


			ImGui::EndMainMenuBar();
		}

		// Bottom Right widget
		{
			std::string rEngineName = engine->name();
			ImGui::Begin("Bottom Left widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::Text("Rendered by %s: %.1f FPS", rEngineName.c_str(), ImGui::GetIO().Framerate);

			ImGui::SetWindowPos(ImVec2(io.DisplaySize.x - ImGui::GetWindowSize().x, io.DisplaySize.y - ImGui::GetWindowSize().y));
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
	
	// draw a rect for selection area 
	drawSelectedRegion();

	// current active node
	drawNodeManipulator(app->currNode, camera->getViewMat(), camera->getProjMat());

	// view manipulator
	if(mViewManipulator)
		drawViewManipulator(camera.get());
	
	// Draw custom widgets
	// gather visual modules
	WidgetQueue imWidgetQueue;
	// enqueue render content
	if (scene && !scene->isEmpty())
	{
		scene->traverseForward(&imWidgetQueue);
	}

	for (auto widget : imWidgetQueue.modules)
	{
		widget->update();
		widget->paint();
	}
}

void ImWindow::mouseMoveEvent(const PMouseEvent& event)
{
	mCurX = event.x;
	mCurY = event.y;
}

void ImWindow::mouseReleaseEvent(const PMouseEvent& event)
{
	// reset mouse status
	mCurX = mRegX = -1;
	mCurY = mRegY = -1;
	mButtonType = BT_UNKOWN;
	mButtonAction = AT_UNKOWN;
	mButtonMode = MB_NO_MODIFIER;
}

void ImWindow::mousePressEvent(const PMouseEvent& event)
{
	mCurX = mRegX = event.x;
	mCurY = mRegY = event.y;
	mButtonAction = event.actionType;
	mButtonType = event.buttonType;
	mButtonMode = event.mods;
}

void ImWindow::drawSelectedRegion()
{
	if (mButtonType == BT_LEFT &&
		mButtonAction == AT_PRESS &&
		mButtonMode == MB_SHIFT &&
		!ImGuizmo::IsUsing() &&
		!ImGui::GetIO().WantCaptureMouse) {

		ImVec2 pMin = { fminf(mRegX, mCurX), fminf(mRegY, mCurY) };
		ImVec2 pMax = { fmaxf(mRegX, mCurX), fmaxf(mRegY, mCurY) };

		// visible rectangle
		if (pMin.x != pMax.x || pMin.y != pMax.y) {
			// fill
			ImGui::GetBackgroundDrawList()->AddRectFilled(pMin, pMax, ImColor{ 0.2f, 0.2f, 0.2f, 0.5f });
			// border
			ImGui::GetBackgroundDrawList()->AddRect(pMin, pMax, ImColor{ 0.8f, 0.8f, 0.8f, 0.8f }, 0, 0, 1.5f);
		}
	}
}

void dyno::ImWindow::drawNodeManipulator(Node* n, glm::mat4 view, glm::mat4 proj)
{
	// TODO: type conversion...
	auto* node = dynamic_cast<ParametricModel<DataType3f>*>(n);

	// TODO: parameterized operation
	auto nodeOp = ImGuizmo::TRANSLATE;

	switch (mEditMode) {
	case 0:
		nodeOp = ImGuizmo::TRANSLATE;
		break;
	case 1:
		nodeOp = ImGuizmo::SCALE;
		break;
	case 2:
		nodeOp = ImGuizmo::ROTATE;
		break;
	default:
		break;
	}

	if (node != 0) {
		float m[16];
		// build transform matrix
		{
			auto t = node->varLocation()->getData();
			auto s = node->varScale()->getData();
			auto r = node->varRotation()->getData();			
			ImGuizmo::RecomposeMatrixFromComponents(t.getDataPtr(), r.getDataPtr(), s.getDataPtr(), m);
		}

		if (ImGuizmo::Manipulate(&view[0][0], &proj[0][0], nodeOp, ImGuizmo::WORLD, m, NULL, NULL, NULL, NULL))
		{
			float t[3], s[3], r[3];
			ImGuizmo::DecomposeMatrixToComponents(m, t, r, s);
			// apply
			if (nodeOp == ImGuizmo::TRANSLATE) 
				node->varLocation()->setValue(Vec3f(t[0], t[1], t[2]));
			if (nodeOp == ImGuizmo::SCALE) 
				node->varScale()->setValue(Vec3f(s[0], s[1], s[2]));
			if (nodeOp == ImGuizmo::ROTATE) 
				node->varRotation()->setValue(Vec3f(r[0], r[1], r[2]));

			// notify the update of node?
			node->update();
		}
	}
}

void dyno::ImWindow::drawViewManipulator(Camera* camera)
{
	glm::mat4 view = camera->getViewMat();
	glm::mat4 view0 = view;
	float dist = (camera->getEyePos() - camera->getTargetPos()).norm();

	ImGuizmo::ViewManipulate(&view[0][0], dist, 
		ImVec2(0, camera->viewportHeight() - 100), 
		ImVec2(100, 100),
		0);

	// if nothing changes
	if (view == view0) return;

	glm::mat4 invView = glm::inverse(view);
	glm::vec4 eye = invView * glm::vec4(0, 0, 0, 1);	

	// for trackball camera, also update up direction
	TrackballCamera* cam = dynamic_cast<TrackballCamera*>(camera);
	if (cam) {
		glm::vec4 up = invView * glm::vec4(0, 1, 0, 0);
		cam->mCameraUp = {up.x, up.y, up.z};

		cam->setEyePos({ eye.x, eye.y, eye.z });
	}
	else
	{
		OrbitCamera* cam = dynamic_cast<OrbitCamera*>(camera);
		// for orbit camera, simply set eye position...
		cam->setEyePos({ eye.x, eye.y, eye.z });

		// TODO: fix the problem when view along the rotation axis
	}
}

bool ImWindow::cameraLocked()
{
	return (ImGui::IsWindowFocused(ImGuiFocusedFlags_::ImGuiFocusedFlags_AnyWindow) || mDisenableCamera);
}