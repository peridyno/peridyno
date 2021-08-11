#include "POpenGLWidget.h"
#include "RenderEngine.h"
#include "RenderTarget.h"
#include "SceneGraph.h"
#include "camera/OrbitCamera.h"
#include "PSimulationThread.h"

//Qt
#include <QMouseEvent>


namespace dyno
{

	POpenGLWidget::POpenGLWidget()
	{
		QSurfaceFormat format;
		format.setDepthBufferSize(24);
		format.setMajorVersion(4);
		format.setMinorVersion(4);
		format.setSamples(4);
		format.setProfile(QSurfaceFormat::CoreProfile);
		setFormat(format);

		// Update at 60 fps
		QObject::connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
		timer.start(16);
	}

	POpenGLWidget::~POpenGLWidget()
	{
		timer.stop();
		delete mRenderEngine;
		delete mRenderTarget;
		delete mRenderParams;
	}

	void POpenGLWidget::initializeGL()
	{


		if (!gladLoadGL()) {
			//SPDLOG_CRITICAL("Failed to load GLAD!");
			exit(-1);
		}

		SceneGraph::getInstance().initialize();

		mRenderEngine = new RenderEngine();
		mRenderTarget = new RenderTarget();
		mRenderParams = new RenderParams();

		mCamera = std::make_shared<OrbitCamera>();
		mCamera->setWidth(this->width());
		mCamera->setHeight(this->height());
		mCamera->registerPoint(0.5f, 0.5f);
		mCamera->translateToPoint(0, 0);

		mCamera->zoom(3.0f);
		mCamera->setClipNear(0.01f);
		mCamera->setClipFar(10.0f);

		mRenderEngine->initialize();
		mRenderTarget->initialize();

		initializeOpenGLFunctions();
		QtImGui::initialize(this);
	}

	void POpenGLWidget::paintGL()
	{
		//QtImGui
		QtImGui::newFrame();
		
		//ImGui
		{
			{// Top Left widget
				ImGui::SetNextWindowPos(ImVec2(0,0));
				ImGui::Begin("Top Left widget",NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
				
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

				// if(CameraType(camera_current) != mCameraType){
					// FIXME: GL error
					// setCameraType(CameraType(camera_current));
				// }

				ImGui::End();
			}	

			{// Top Right widget
				
				ImGui::Begin("Top Right widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
				ImGui::toggleButton("Lock", &(mLock));
				ImGui::SameLine();
				ImGui::toggleButton("Ground", &(mRenderParams->showGround));
				ImGui::SameLine();
				ImGui::toggleButton("Bounds",&(mRenderParams->showSceneBounds));
				ImGui::SameLine();
				ImGui::toggleButton("Axis Helper", &(mRenderParams->showAxisHelper));
				ImGui::SetWindowPos(ImVec2(width() - ImGui::GetWindowSize().x, 0));

				ImGui::End();
			}

			{// Bottom Right widget
				ImGui::Begin("Bottom Left widget", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
				ImGui::Text(" %.1f FPS", ImGui::GetIO().Framerate);
				ImGui::SetWindowPos(ImVec2(width() - ImGui::GetWindowSize().x, height() - ImGui::GetWindowSize().y));
				ImGui::End();
			}
			mOpenCameraRotate = !(ImGui::IsWindowFocused(ImGuiFocusedFlags_::ImGuiFocusedFlags_AnyWindow) || mLock);
		}

		ImGui::Render();
		// Do render After ImGui UI is rendered
        glViewport(0, 0, width(), height());
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		// Graphscrene draw
		GLint fbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);

		mRenderParams->proj = mCamera->getProjMat();
		mRenderParams->view = mCamera->getViewMat();

		mRenderEngine->draw(&SceneGraph::getInstance(), mRenderTarget, *mRenderParams);

		// write back to the framebuffer
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
		mRenderTarget->blit(0);
        
		// Do QtImgui Render After Glfw Render
		QtImGui::render();
	}

	void POpenGLWidget::resizeGL(int w, int h)
	{
		mCamera->setWidth(w);
		mCamera->setHeight(h);

		mRenderTarget->resize(w, h);
		// set the viewport
		mRenderParams->viewport.x = 0;
		mRenderParams->viewport.y = 0;
		mRenderParams->viewport.w = w;
		mRenderParams->viewport.h = h;
	}

	void POpenGLWidget::mousePressEvent(QMouseEvent *event)
	{
		mCamera->registerPoint(event->x(), event->y());
		mButtonState = QButtonState::QBUTTON_DOWN;
	}

	void POpenGLWidget::mouseReleaseEvent(QMouseEvent *event)
	{
		mButtonState = QButtonState::QBUTTON_UP;
	}

	void POpenGLWidget::mouseMoveEvent(QMouseEvent *event)
	{
		if (event->buttons().testFlag(Qt::LeftButton) && mButtonState == QBUTTON_DOWN && mOpenCameraRotate) {
			mCamera->rotateToPoint(event->x(), event->y());
		}
		else if (event->buttons().testFlag(Qt::RightButton) && mButtonState == QBUTTON_DOWN && mOpenCameraRotate) {
			mCamera->translateToPoint(event->x(), event->y());
		}

		update();
	}

	void POpenGLWidget::wheelEvent(QWheelEvent *event)
	{
		if(mOpenCameraRotate)mCamera->zoom(-0.001*event->angleDelta().y());
		update();
	}

	void POpenGLWidget::updateGraphicsContext()
	{
		PSimulationThread::instance()->startRendering();
		
		SceneGraph::getInstance().updateGraphicsContext();
		update();

		PSimulationThread::instance()->stopRendering();
	}

}