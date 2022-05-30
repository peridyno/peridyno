#include <glad/glad.h>

#include "POpenGLWidget.h"
#include "PSimulationThread.h"

#include <Rendering.h>
#include <SceneGraph.h>
#include <OrbitCamera.h>

//Qt
#include <QMouseEvent>
#include <QGuiApplication>
#include <QScreen>

#include "QtImGui.h"
#include <ImWidget.h>

#include "SceneGraphFactory.h"

namespace dyno
{

	POpenGLWidget::POpenGLWidget(RenderEngine* engine)
	{
		QSurfaceFormat format;
		format.setDepthBufferSize(24);
		format.setMajorVersion(4);
		format.setMinorVersion(4);
		format.setSamples(4);
		format.setSwapInterval(1);
		format.setProfile(QSurfaceFormat::CoreProfile);
		setFormat(format);

		// Update at 60 fps
		QObject::connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
		timer.start(16);

		mRenderEngine = engine;
	}

	POpenGLWidget::~POpenGLWidget()
	{
		timer.stop();
		//delete mRenderEngine;
	}

	void POpenGLWidget::initializeGL()
	{
		if (!gladLoadGL()) {
			//SPDLOG_CRITICAL("Failed to load GLAD!");
			exit(-1);
		}

		auto scn = SceneGraphFactory::instance()->active();

		scn->initialize();

		initializeOpenGLFunctions();
		QtImGui::initialize(this);

		//mRenderEngine = new RenderEngine();
		// Get Context scale
		float scale = QGuiApplication::primaryScreen()->logicalDotsPerInchX() / 96.0;
		mRenderEngine->initialize(this->width(), this->height());

		mImWindow.initialize(scale);
	}

	void POpenGLWidget::paintGL()
	{
		//QtImGui
		QtImGui::newFrame();
		
		auto scn = SceneGraphFactory::instance()->active();
		// Draw scene		
		mRenderEngine->draw(scn.get());

		// Draw ImGui
		mRenderEngine->renderParams()->viewport.w = this->width();
		mRenderEngine->renderParams()->viewport.h = this->height();
		mImWindow.draw(mRenderEngine, scn.get());
		// Draw widgets
// 		// TODO: maybe move into mImWindow...
// 		for (auto widget : mWidgets)
// 		{
// 			widget->update();
// 			widget->paint();
// 		}

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
		if (event->buttons().testFlag(Qt::LeftButton) && mButtonState == QBUTTON_DOWN && !mImWindow.cameraLocked()) {
			activeCamera()->rotateToPoint(event->x(), event->y());
		}
		else if (event->buttons().testFlag(Qt::RightButton) && mButtonState == QBUTTON_DOWN && !mImWindow.cameraLocked()) {
			activeCamera()->translateToPoint(event->x(), event->y());
		}

		update();
	}

	void POpenGLWidget::wheelEvent(QWheelEvent *event)
	{
		if(!mImWindow.cameraLocked())
			activeCamera()->zoom(-0.001*event->angleDelta().y());

		update();
	}

	void POpenGLWidget::updateGraphicsContext()
	{
		makeCurrent();

		PSimulationThread::instance()->startRendering();
		
		//Update SceneGraph
		SceneGraphFactory::instance()->active()->updateGraphicsContext();

		//Update QtWidget
		this->update();

		PSimulationThread::instance()->stopRendering();

		doneCurrent();
	}

	std::shared_ptr<Camera> POpenGLWidget::activeCamera()
	{
		return mRenderEngine->camera();
	}

}