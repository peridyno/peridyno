#include "GLRenderEngine.h"
#include "GLRenderHelper.h"
#include "GLRenderTarget.h"
#include "GLVisualModule.h"

#include "Utility.h"
#include "ShadowMap.h"
#include "SSAO.h"

#include "ui/imgui_extend.h"

// dyno
#include "SceneGraph.h"
#include "Action.h"

// GLM
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>

#include <OrbitCamera.h>
#include <TrackballCamera.h>

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

	GLRenderEngine::GLRenderEngine()
	{
		mRenderHelper = new GLRenderHelper();
		mShadowMap = new ShadowMap(2048, 2048);
		mSSAO = new SSAO;

		setupCamera();
	}

	GLRenderEngine::~GLRenderEngine()
	{
		delete mRenderHelper;
		delete mShadowMap;
		delete mSSAO;
	}

	void GLRenderEngine::initialize(int width, int height, float scale)
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

		mRenderTarget = new GLRenderTarget();

		mRenderTarget->initialize();
		mRenderTarget->resize(width, height);

		m_camera->setWidth(width);
		m_camera->setHeight(height);
		m_camera->registerPoint(0.5f, 0.5f);
		m_camera->translateToPoint(0, 0);

		m_camera->zoom(3.0f);
		m_camera->setClipNear(0.01f);
		m_camera->setClipFar(10.0f);

		// TODO: Reorganize
		mPics.emplace_back(std::make_shared<Picture>("../../data/icon/map.png"));
		mPics.emplace_back(std::make_shared<Picture>("../../data/icon/box.png"));
		mPics.emplace_back(std::make_shared<Picture>("../../data/icon/arrow-090-medium.png"));
		mPics.emplace_back(std::make_shared<Picture>("../../data/icon/lock.png"));		

		ImGui::initializeStyle(scale);
	}

	void GLRenderEngine::setupCamera()
	{
		switch (mCameraType)
		{
		case dyno::Orbit:
			m_camera = std::make_shared<OrbitCamera>();
			break;
		case dyno::TrackBall:
			m_camera = std::make_shared<TrackballCamera>();
			break;
		default:
			break;
		}
	}

	void GLRenderEngine::initUniformBuffers()
	{
		// create uniform block for transform
		mTransformUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		// create uniform block for light
		mLightUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		gl::glCheckError();
	}

	void GLRenderEngine::draw(dyno::SceneGraph* scene)
	{
		m_rparams.proj = m_camera->getProjMat();
		m_rparams.view = m_camera->getViewMat();

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
		mShadowMap->update(scene, m_rparams);

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
		sceneUniformBuffer.view = m_rparams.view;
		sceneUniformBuffer.projection = m_rparams.proj;
		sceneUniformBuffer.width = mRenderTarget->width;
		sceneUniformBuffer.height = mRenderTarget->height;

		mTransformUBO.load(&sceneUniformBuffer, sizeof(sceneUniformBuffer));
		mTransformUBO.bindBufferBase(0);

		// setup light block
		RenderParams::Light light = m_rparams.light;
		light.mainLightDirection = glm::vec3(m_rparams.view * glm::vec4(light.mainLightDirection, 0));
		mLightUBO.load(&light, sizeof(light));
		mLightUBO.bindBufferBase(1);

		// begin rendering
		mRenderTarget->bind();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		Vec3f c0 = Vec3f(m_rparams.bgColor0.x, m_rparams.bgColor0.y, m_rparams.bgColor0.z);
		Vec3f c1 = Vec3f(m_rparams.bgColor1.x, m_rparams.bgColor1.y, m_rparams.bgColor1.z);
		mRenderHelper->drawBackground(c0, c1);

		glClear(GL_DEPTH_BUFFER_BIT);
		// draw a plane
		if (m_rparams.showGround)
		{
			mRenderHelper->drawGround(m_rparams.groudScale);
		}

		// render modules
		for (GLVisualModule* m : renderQueue.modules)
		{
			m->paintGL(GLVisualModule::COLOR);
		}

		// draw scene bounding box
		if (m_rparams.showSceneBounds && scene != 0)
		{
			// get bounding box of the scene
			auto p0 = scene->getLowerBound();
			auto p1 = scene->getUpperBound();
			mRenderHelper->drawBBox(p0, p1);
		}
		// draw axis
		if (m_rparams.showAxisHelper)
		{
			glViewport(10, 10, 100, 100);
			mRenderHelper->drawAxis();
		}

		// write back to the framebuffer
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
		mRenderTarget->blit(0);
	}

	void GLRenderEngine::drawGUI()
	{
		
		float iBgGray[2] = { m_rparams.bgColor0[0], m_rparams.bgColor1[0]};
		// float iBgGray[2] = { 0.2f, 0.8f };
		RenderParams::Light iLight =m_rparams.light;

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
						m_rparams.bgColor0 = glm::vec3(iBgGray[0]);
						m_rparams.bgColor1 = glm::vec3(iBgGray[1]);
						
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
						m_rparams.light = iLight;

						ImGui::EndPopup();
					}
					

					// Camera Select
					static int camera_current = 0;
					const char* camera_name[] = {"Orbit", "TrackBall"};
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
					ImGui::toggleButton(mPics[3]->GetTexture(),"Lock", &(mDisenableCamera));
					// ImGui::toggleButton("Lock", &(mDisenableCamera));
					ImGui::SameLine();
					ImGui::toggleButton(mPics[0]->GetTexture(),"Ground", &(m_rparams.showGround));
					// ImGui::toggleButton("Ground", &(m_rparams.showGround));
					ImGui::SameLine();
					ImGui::toggleButton(mPics[1]->GetTexture(),"Bounds", &(m_rparams.showSceneBounds));
					// ImGui::toggleButton("Bounds", &(m_rparams.showSceneBounds));
					ImGui::SameLine();
					ImGui::toggleButton(mPics[2]->GetTexture(),"Axis Helper", &(m_rparams.showAxisHelper));
					// ImGui::toggleButton("Axis Helper", &(m_rparams.showAxisHelper));
					ImGui::SetWindowPos(ImVec2(mRenderTarget->width - ImGui::GetWindowSize().x, 0));

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

	void GLRenderEngine::resize(int w, int h)
	{
		mRenderTarget->resize(w, h);
		// set the viewport
		m_rparams.viewport.x = 0;
		m_rparams.viewport.y = 0;
		m_rparams.viewport.w = w;
		m_rparams.viewport.h = h;
	}

	bool GLRenderEngine::cameraLocked()
	{
		return (ImGui::IsWindowFocused(ImGuiFocusedFlags_::ImGuiFocusedFlags_AnyWindow) || mDisenableCamera);
	}

}