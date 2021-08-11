#include "RenderEngine.h"
// dyno
#include "SceneGraph.h"
#include "Action.h"

// GLM
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>

#include "ui/imgui_extend.h"

#include "Utility.h"
#include "ShadowMap.h"
#include "SSAO.h"
#include "RenderHelper.h"
#include "RenderTarget.h"

#include "module/GLVisualModule.h"

#include "camera/OrbitCamera.h"
#include "camera/TrackballCamera.h"

namespace dyno
{
	class RenderQueue : public Action
	{
	private:
		void process(Node* node) override
		{
			if (!node->isVisible())
				return;

			for (auto iter : node->graphicsPipeline()->activeModules())
			{
				auto m = dynamic_cast<GLVisualModule*>(iter);
				if (m && m->isVisible())
				{
					//m->update();
					modules.push_back(m);
				}
			}
		}
	public:
		std::vector<dyno::GLVisualModule*> modules;
	};

	RenderEngine::RenderEngine()
	{
		mRenderHelper = new RenderHelper();
		mShadowMap = new ShadowMap(2048, 2048);
		mSSAO = new SSAO;

		setupCamera();
	}

	RenderEngine::~RenderEngine()
	{
		delete mRenderHelper;
		delete mShadowMap;
		delete mSSAO;
	}

	void RenderEngine::initialize(int width, int height)
	{
		if (!gladLoadGL()) {
			printf("Failed to load OpenGL!");
			exit(-1);
		}

		// some basic opengl settings
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glDepthFunc(GL_LEQUAL);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		initUniformBuffers();

		mSSAO->initialize();
		mShadowMap->initialize();
		mRenderHelper->initialize();

		mRenderTarget = new RenderTarget();
		mRenderParams = new RenderParams();

		mRenderTarget->initialize();
		mRenderTarget->resize(width, height);

		mCamera->setWidth(width);
		mCamera->setHeight(height);
		mCamera->registerPoint(0.5f, 0.5f);
		mCamera->translateToPoint(0, 0);

		mCamera->zoom(3.0f);
		mCamera->setClipNear(0.01f);
		mCamera->setClipFar(10.0f);

		// TODO: Reorganize
		mPics.emplace_back(std::make_shared<Picture>("../../data/icon/map.png"));
		mPics.emplace_back(std::make_shared<Picture>("../../data/icon/box.png"));
		mPics.emplace_back(std::make_shared<Picture>("../../data/icon/arrow-090-medium.png"));
		mPics.emplace_back(std::make_shared<Picture>("../../data/icon/lock.png"));		
	}

	void RenderEngine::setupCamera()
	{
		switch (mCameraType)
		{
		case dyno::Orbit:
			mCamera = std::make_shared<OrbitCamera>();
			break;
		case dyno::TrackBall:
			mCamera = std::make_shared<TrackballCamera>();
			break;
		default:
			break;
		}
	}

	void RenderEngine::begin()
	{
		glViewport(0, 0, mRenderTarget->width, mRenderTarget->height);
		glClearColor(mClearColor.x * mClearColor.w, mClearColor.y * mClearColor.w, mClearColor.z * mClearColor.w, mClearColor.w);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	void RenderEngine::end()
	{

	}

	void RenderEngine::initUniformBuffers()
	{
		// create uniform block for transform
		mTransformUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		// create uniform block for light
		mLightUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		gl::glCheckError();
	}

	void RenderEngine::draw(dyno::SceneGraph* scene)
	{
		mRenderParams->proj = mCamera->getProjMat();
		mRenderParams->view = mCamera->getViewMat();

		// Graphscrene draw
		GLint fbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);

		// gather visual modules
		RenderQueue renderQueue;
		// enqueue render content
		if ((scene != 0) && (scene->getRootNode() != 0))
		{
			scene->getRootNode()->traverseTopDown(&renderQueue);
		}

		// update shadow map
		mShadowMap->update(scene, *mRenderParams);

		// setup scene transform matrices
		struct
		{
			glm::mat4 model;
			glm::mat4 view;
			glm::mat4 projection;
			int width;
			int height;
		} sceneUniformBuffer;
		sceneUniformBuffer.model = glm::mat4(1);
		sceneUniformBuffer.view = mRenderParams->view;
		sceneUniformBuffer.projection = mRenderParams->proj;
		sceneUniformBuffer.width = mRenderTarget->width;
		sceneUniformBuffer.height = mRenderTarget->height;

		mTransformUBO.load(&sceneUniformBuffer, sizeof(sceneUniformBuffer));
		mTransformUBO.bindBufferBase(0);

		// setup light block
		RenderParams::Light light = mRenderParams->light;
		light.mainLightDirection = glm::vec3(mRenderParams->view * glm::vec4(light.mainLightDirection, 0));
		mLightUBO.load(&light, sizeof(light));
		mLightUBO.bindBufferBase(1);

		// begin rendering
		mRenderTarget->bind();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		Vec3f c0 = Vec3f(mRenderParams->bgColor0.x, mRenderParams->bgColor0.y, mRenderParams->bgColor0.z);
		Vec3f c1 = Vec3f(mRenderParams->bgColor1.x, mRenderParams->bgColor1.y, mRenderParams->bgColor1.z);
		mRenderHelper->drawBackground(c0, c1);

		glClear(GL_DEPTH_BUFFER_BIT);
		// draw a plane
		if (mRenderParams->showGround)
		{
			mRenderHelper->drawGround(mRenderParams->groudScale);
		}


		//glBegin(GL_TRIANGLES);

		//glVertex3f(0.0f, 1.0f, 0.0f); glColor3f(1.0f, 0.0f, 0.0f);

		//glVertex3f(-1.0f, 0.0f, 0.0f); glColor3f(0.0f, 1.0f, 0.0f);

		//glVertex3f(1.0f, 0.0f, 0.0f); glColor3f(0.0f, 0.0f, 1.0f);
		//glEnd();

		// render modules
		for (GLVisualModule* m : renderQueue.modules)
		{
			m->paintGL(GLVisualModule::COLOR);
		}

		// draw scene bounding box
		if (mRenderParams->showSceneBounds && scene != 0)
		{
			// get bounding box of the scene
			auto p0 = scene->getLowerBound();
			auto p1 = scene->getUpperBound();
			mRenderHelper->drawBBox(p0, p1);
		}
		// draw axis
		if (mRenderParams->showAxisHelper)
		{
			glViewport(10, 10, 100, 100);
			mRenderHelper->drawAxis();
		}


		// write back to the framebuffer
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
		mRenderTarget->blit(0);
	}

	void RenderEngine::drawGUI()
	{
		float iBgGray[2] = { 0.2f, 0.8f };
		RenderParams::Light iLight;

 			// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
 			{
				static float f = 0.0f;
				static int counter = 0;

				{// Top Left widget
					ImGui::SetNextWindowPos(ImVec2(0,0));
					ImGui::Begin("Top Left widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
					
					if(ImGui::Button("Lighting")){
						ImGui::OpenPopup("LightingMenu");
					}

					if(ImGui::BeginPopup("LightingMenu")){
						ImGui::SliderFloat2("BG color", iBgGray, 0.0f, 1.0f, "%.3f", 0);
						mRenderParams->bgColor0 = glm::vec3(iBgGray[0]);
						mRenderParams->bgColor1 = glm::vec3(iBgGray[1]);
						
						ImGui::Text("Ambient Light");

						ImGui::beginTitle("Ambient Light Scale");
						ImGui::SliderFloat("", &iLight.ambientScale, 0.0f, 10.0f, "%.3f", 0); 
						ImGui::endTitle();
						ImGui::SameLine();
						ImGui::ColorEdit3("Ambient Light Color", (float*)&iLight.ambientColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float | ImGuiColorEditFlags_NoDragDrop | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoLabel) ;

						ImGui::Text("Main Light");
						ImGui::beginTitle("Main Light Scale");
						ImGui::SliderFloat("", &iLight.mainLightScale, 0.0f, 10.0f, "%.3f", 0); 
						ImGui::endTitle();
						ImGui::SameLine();
						ImGui::ColorEdit3("Main Light Color", (float*)&iLight.mainLightColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float | ImGuiColorEditFlags_NoDragDrop | ImGuiColorEditFlags_AlphaPreview | ImGuiColorEditFlags_NoLabel);
						mRenderParams->light = iLight;

						ImGui::EndPopup();
					}
					

					// Camera Select
					static int camera_current = 0;
					const char* camera_name[] = {"Orbit", "TrackBall"};
					static ImGuiComboFlags flags = ImGuiComboFlags_NoArrowButton;
					// ImGui::Combo("Camera", &camera_current, camera_name, IM_ARRAYSIZE(camera_name));
					ImGui::SetNextItemWidth(100);

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
// 						// FIXME: GL error
// 						// setCameraType(CameraType(camera_current));
// 					}
					ImGui::End();
				}

				{// Top Right widget
					
					ImGui::Begin("Top Right widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
					ImGui::toggleButton(mPics[3]->GetTexture(),"Lock", &(mDisenableCamera));
					// ImGui::toggleButton("Lock", &(mDisenableCamera));
					ImGui::SameLine();
					ImGui::toggleButton(mPics[0]->GetTexture(),"Ground", &(mRenderParams->showGround));
					// ImGui::toggleButton("Ground", &(mRenderParams->showGround));
					ImGui::SameLine();
					ImGui::toggleButton(mPics[1]->GetTexture(),"Bounds",&(mRenderParams->showSceneBounds));
					// ImGui::toggleButton("Bounds", &(mRenderParams->showSceneBounds));
					ImGui::SameLine();
					ImGui::toggleButton(mPics[2]->GetTexture(),"Axis Helper", &(mRenderParams->showAxisHelper));
					// ImGui::toggleButton("Axis Helper", &(mRenderParams->showAxisHelper));
					ImGui::SetWindowPos(ImVec2(mRenderTarget->width - ImGui::GetWindowSize().x, 0));

					ImGui::End();
				}

				{// Right sidebar
					ImGui::Begin("Right sidebar", NULL, /*ImGuiWindowFlags_NoMove |*/ ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
						
					const int val[6 + 1] = {0,1,2,3,4,5,6};
					const int style_alpha8 = 150;
					const ImU32 col[6 + 1] = { IM_COL32(255,0,0,style_alpha8), IM_COL32(255,255,0,style_alpha8), IM_COL32(0,255,0,style_alpha8), IM_COL32(0,255,255,style_alpha8), IM_COL32(0,0,255,style_alpha8), IM_COL32(255,0,255,style_alpha8), IM_COL32(255,0,0,style_alpha8) };
					ImGui::ColorBar("ColorBar", val, col, 7);

					// ImGui::SetWindowPos(ImVec2(mRenderTarget->width - ImGui::GetWindowSize().x, (mRenderTarget->height - ImGui::GetWindowSize().y) /2));
					ImGui::End();
				}


				{// Bottom Right widget
					ImGui::Begin("Bottom Left widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
					ImGui::Text(" %.1f FPS", ImGui::GetIO().Framerate);
					ImGui::SetWindowPos(ImVec2(mRenderTarget->width - ImGui::GetWindowSize().x, mRenderTarget->height - ImGui::GetWindowSize().y));
					ImGui::End();
				}
 			}

	}

	void RenderEngine::resizeRenderTarget(int w, int h)
	{
		mRenderTarget->resize(w, h);
		// set the viewport
		mRenderParams->viewport.x = 0;
		mRenderParams->viewport.y = 0;
		mRenderParams->viewport.w = w;
		mRenderParams->viewport.h = h;
	}

	bool RenderEngine::cameraLocked()
	{
		// return !mOpenCameraRotate;
		return (ImGui::IsWindowFocused(ImGuiFocusedFlags_::ImGuiFocusedFlags_AnyWindow) || mDisenableCamera);
	}

	// //TODO: 
	// void RenderEngine::loadIcon() {
	// 	mPics.emplace_back(std::make_shared<Picture>("../../data/icon/map.png"));
	// 	mPics.emplace_back(std::make_shared<Picture>("../../data/icon/box.png"));
	// 	mPics.emplace_back(std::make_shared<Picture>("../../data/icon/arrow-090-medium.png"));
	// 	mPics.emplace_back(std::make_shared<Picture>("../../data/icon/lock.png"));
	// }
}