#include "POpenGLWidget.h"
#include "RenderEngine.h"
#include "RenderTarget.h"
#include "RenderParams.h"
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
	}

	POpenGLWidget::~POpenGLWidget()
	{
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
		QtImGUI::initialize(this);
	}

	void POpenGLWidget::paintGL()
	{
		QtImGUI::newFrame();

		{
			static float f = 0.0f;
			ImGui::Text("Hello, world!");
			ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
		}

		// Graphscrene draw
		GLint fbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);

		mRenderParams->proj = mCamera->getProjMat();
		mRenderParams->view = mCamera->getViewMat();

		mRenderEngine->draw(&SceneGraph::getInstance(), mRenderTarget, *mRenderParams);

		// write back to the framebuffer
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
		mRenderTarget->blit(0);

		ImGui::Render();
		QtImGUI::render();
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
		if (event->buttons().testFlag(Qt::LeftButton) && mButtonState == QBUTTON_DOWN) {
			mCamera->rotateToPoint(event->x(), event->y());
		}
		else if (event->buttons().testFlag(Qt::RightButton) && mButtonState == QBUTTON_DOWN) {
			mCamera->translateToPoint(event->x(), event->y());
		}

		update();
	}

	void POpenGLWidget::wheelEvent(QWheelEvent *event)
	{
		mCamera->zoom(-0.001*event->angleDelta().y());
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