#include "POpenGLWidget.h"
#include "RenderEngine.h"
#include "RenderTarget.h"
#include "SceneGraph.h"
#include "camera/OrbitCamera.h"
#include "PSimulationThread.h"

//Qt
#include <QMouseEvent>
#include <QGuiApplication>
#include <QScreen>


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
	}

	void POpenGLWidget::initializeGL()
	{
		if (!gladLoadGL()) {
			//SPDLOG_CRITICAL("Failed to load GLAD!");
			exit(-1);
		}

		SceneGraph::getInstance().initialize();

		initializeOpenGLFunctions();
		QtImGui::initialize(this);

		mRenderEngine = new RenderEngine();
		// Get Context scale
		float scale = QGuiApplication::primaryScreen()->logicalDotsPerInchX() / 96.0;
		mRenderEngine->initialize(this->width(), this->height(), scale);
	}

	void POpenGLWidget::paintGL()
	{
		//QtImGui
		QtImGui::newFrame();
		
		mRenderEngine->begin();
		
		mRenderEngine->draw(&SceneGraph::getInstance());
        
		mRenderEngine->drawGUI();

		mRenderEngine->end();

		ImGui::Render();
		// Do QtImgui Render After Glfw Render
		QtImGui::render();
	}

	void POpenGLWidget::resizeGL(int w, int h)
	{
		activeCamera()->setWidth(w);
		activeCamera()->setHeight(h);

		mRenderEngine->resize(w, h);
	}

	void POpenGLWidget::mousePressEvent(QMouseEvent *event)
	{
		activeCamera()->registerPoint(event->x(), event->y());
		mButtonState = QButtonState::QBUTTON_DOWN;
	}

	void POpenGLWidget::mouseReleaseEvent(QMouseEvent *event)
	{
		mButtonState = QButtonState::QBUTTON_UP;
	}

	void POpenGLWidget::mouseMoveEvent(QMouseEvent *event)
	{
		if (event->buttons().testFlag(Qt::LeftButton) && mButtonState == QBUTTON_DOWN && !mRenderEngine->cameraLocked()) {
			activeCamera()->rotateToPoint(event->x(), event->y());
		}
		else if (event->buttons().testFlag(Qt::RightButton) && mButtonState == QBUTTON_DOWN && !mRenderEngine->cameraLocked()) {
			activeCamera()->translateToPoint(event->x(), event->y());
		}

		update();
	}

	void POpenGLWidget::wheelEvent(QWheelEvent *event)
	{
		if(!mRenderEngine->cameraLocked())
			activeCamera()->zoom(-0.001*event->angleDelta().y());

		update();
	}

	void POpenGLWidget::updateGraphicsContext()
	{
		PSimulationThread::instance()->startRendering();
		
		SceneGraph::getInstance().updateGraphicsContext();
		update();

		PSimulationThread::instance()->stopRendering();
	}

	std::shared_ptr<Camera> POpenGLWidget::activeCamera()
	{
		return mRenderEngine->camera();
	}

}