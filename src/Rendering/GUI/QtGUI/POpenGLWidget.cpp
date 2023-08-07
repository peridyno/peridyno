#include <glad/glad.h>

#include "POpenGLWidget.h"
#include "PSimulationThread.h"

#include <SceneGraph.h>
#include <OrbitCamera.h>

//Qt
#include <QMouseEvent>
#include <QGuiApplication>
#include <QScreen>

#include "QtImGui.h"
#include <ImWidget.h>

#include "SceneGraphFactory.h"

#include <GLRenderEngine.h>
#include "QtApp.h"
#include "ImGuizmo.h"

#include <map>

namespace dyno
{
	std::map<int, PKeyboardType> KeyMap =
	{
		{Qt::Key_unknown, PKEY_UNKNOWN},
		{Qt::Key_Space, PKEY_SPACE},
		{Qt::Key_Apostrophe, PKEY_APOSTROPHE},
		{Qt::Key_Comma, PKEY_COMMA},
		{Qt::Key_Minus, PKEY_MINUS},
		{Qt::Key_Period, PKEY_PERIOD},
		{Qt::Key_Slash, PKEY_SLASH},
		{Qt::Key_0, PKEY_0},
		{Qt::Key_1, PKEY_1},
		{Qt::Key_2, PKEY_2},
		{Qt::Key_3, PKEY_3},
		{Qt::Key_4, PKEY_4},
		{Qt::Key_5, PKEY_5},
		{Qt::Key_6, PKEY_6},
		{Qt::Key_7, PKEY_7},
		{Qt::Key_8, PKEY_8},
		{Qt::Key_9, PKEY_9},
		{Qt::Key_Semicolon, PKEY_SEMICOLON},
		{Qt::Key_Equal, PKEY_EQUAL},
		{Qt::Key_A, PKEY_A},
		{Qt::Key_B, PKEY_B},
		{Qt::Key_C, PKEY_C},
		{Qt::Key_D, PKEY_D},
		{Qt::Key_E, PKEY_E},
		{Qt::Key_F, PKEY_F},
		{Qt::Key_G, PKEY_G},
		{Qt::Key_H, PKEY_H},
		{Qt::Key_I, PKEY_I},
		{Qt::Key_J, PKEY_J},
		{Qt::Key_K, PKEY_K},
		{Qt::Key_L, PKEY_L},
		{Qt::Key_M, PKEY_M},
		{Qt::Key_N, PKEY_N},
		{Qt::Key_O, PKEY_O},
		{Qt::Key_P, PKEY_P},
		{Qt::Key_Q, PKEY_Q},
		{Qt::Key_R, PKEY_R},
		{Qt::Key_S, PKEY_S},
		{Qt::Key_T, PKEY_T},
		{Qt::Key_U, PKEY_U},
		{Qt::Key_V, PKEY_V},
		{Qt::Key_W, PKEY_W},
		{Qt::Key_X, PKEY_X},
		{Qt::Key_Y, PKEY_Y},
		{Qt::Key_Z, PKEY_Z},
		{Qt::Key_Backslash, PKEY_BACKSLASH},
		{Qt::Key_Escape, PKEY_ESCAPE},
		{Qt::Key_Enter, PKEY_ENTER},
		{Qt::Key_Tab, PKEY_TAB},
		{Qt::Key_Backspace, PKEY_BACKSPACE},
		{Qt::Key_Insert, PKEY_INSERT},
		{Qt::Key_Delete, PKEY_DELETE},
		{Qt::Key_Right, PKEY_RIGHT},
		{Qt::Key_Left, PKEY_LEFT},
		{Qt::Key_Down, PKEY_DOWN},
		{Qt::Key_Up, PKEY_UP},
		{Qt::Key_PageUp, PKEY_PAGE_UP},
		{Qt::Key_PageDown, PKEY_PAGE_DOWN},
		{Qt::Key_Home, PKEY_HOME},
		{Qt::Key_End, PKEY_END},
		{Qt::Key_CapsLock, PKEY_CAPS_LOCK},
		{Qt::Key_ScrollLock, PKEY_SCROLL_LOCK},
		{Qt::Key_NumLock, PKEY_NUM_LOCK},
		{Qt::Key_Pause, PKEY_PAUSE},
		{Qt::Key_F1, PKEY_F1},
		{Qt::Key_F2, PKEY_F2},
		{Qt::Key_F3, PKEY_F3},
		{Qt::Key_F4, PKEY_F4},
		{Qt::Key_F5, PKEY_F5},
		{Qt::Key_F6, PKEY_F6},
		{Qt::Key_F7, PKEY_F7},
		{Qt::Key_F8, PKEY_F8},
		{Qt::Key_F9, PKEY_F9},
		{Qt::Key_F10, PKEY_F10},
		{Qt::Key_F11, PKEY_F11},
		{Qt::Key_F12, PKEY_F12},
		{Qt::Key_F13, PKEY_F13},
		{Qt::Key_F14, PKEY_F14},
		{Qt::Key_F15, PKEY_F15},
		{Qt::Key_F16, PKEY_F16},
		{Qt::Key_F17, PKEY_F17},
		{Qt::Key_F18, PKEY_F18},
		{Qt::Key_F19, PKEY_F19},
		{Qt::Key_F20, PKEY_F20},
		{Qt::Key_F21, PKEY_F21},
		{Qt::Key_F22, PKEY_F22},
		{Qt::Key_F23, PKEY_F23},
		{Qt::Key_F24, PKEY_F24},
		{Qt::Key_F25, PKEY_F25}
	};

	POpenGLWidget::POpenGLWidget(QWidget* parent)
		: RenderWindow()
		, QOpenGLWidget(parent)
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

		setFocusPolicy(Qt::StrongFocus);
	}

	POpenGLWidget::~POpenGLWidget()
	{
		timer.stop();
		//delete mRenderEngine;
		
		makeCurrent();
		this->getRenderEngine()->terminate();
		doneCurrent();
	}

	void POpenGLWidget::initializeGL()
	{
		if (!gladLoadGL()) {
			//SPDLOG_CRITICAL("Failed to load GLAD!");
			exit(-1);
		}

		initializeOpenGLFunctions();
		QtImGui::initialize(this);

		// initialize render engine
		if (this->getRenderEngine() == 0) {
			auto engine = std::make_shared<GLRenderEngine>();
			this->setRenderEngine(engine);
			engine->initialize();
		}

		// Get Context scale
		float scale = QGuiApplication::primaryScreen()->logicalDotsPerInchX() / 96.0;
		mImWindow.initialize(scale);

		auto scn = SceneGraphFactory::instance()->active();

		if (scn != nullptr)
		{
			scn->reset();
			scn->updateGraphicsContext();
		}
	}

	void POpenGLWidget::paintGL()
	{
		//QtImGui
		QtImGui::newFrame();

		// update rendering params
		mRenderParams.width = mCamera->viewportWidth();
		mRenderParams.height = mCamera->viewportHeight();
		mRenderParams.transforms.model = glm::mat4(1);	 // TODO: world transform?
		mRenderParams.transforms.view = mCamera->getViewMat();
		mRenderParams.transforms.proj = mCamera->getProjMat();

		// Jian SHI: hack for unit scaling...
		float planeScale = mRenderEngine->planeScale;
		float rulerScale = mRenderEngine->rulerScale;
		mRenderEngine->planeScale *= mCamera->unitScale();
		mRenderEngine->rulerScale *= mCamera->unitScale();

		// Draw scene		
		mRenderEngine->draw(SceneGraphFactory::instance()->active().get(), mRenderParams);

		mRenderEngine->planeScale = planeScale;
		mRenderEngine->rulerScale = rulerScale;

		// Draw ImGui
		mImWindow.draw(this);
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
		qreal ratio = devicePixelRatio();

		this->setWindowSize(floor(w * ratio), floor(h * ratio));
	}

	PButtonType mappingMouseButton(QMouseEvent* event)
	{
		if (event->buttons().testFlag(Qt::LeftButton))
		{
			return PButtonType::BT_LEFT;
		}
		else if (event->buttons().testFlag(Qt::MiddleButton))
		{
			return PButtonType::BT_MIDDLE;
		}
		else if (event->buttons().testFlag(Qt::RightButton))
		{
			return PButtonType::BT_RIGHT;
		}
	}

	PModifierBits mappingModifierBits(Qt::KeyboardModifiers mods)
	{
		if (mods == Qt::ControlModifier)
		{
			return PModifierBits::MB_CONTROL;
		}
		else if (mods == Qt::ShiftModifier)
		{
			return PModifierBits::MB_SHIFT;
		}
		else if (mods == Qt::AltModifier)
		{
			return PModifierBits::MB_ALT;
		}
		else
			return PModifierBits::MB_NO_MODIFIER;
	}

	void POpenGLWidget::mousePressEvent(QMouseEvent *event)
	{
		mButtonState = QButtonState::QBUTTON_DOWN;
		mCursorX = event->x();
		mCursorY = event->y();

		auto camera = this->getCamera();
		camera->registerPoint(event->x(), event->y());

		PMouseEvent mouseEvent;
		mouseEvent.ray = camera->castRayInWorldSpace((float)event->x(), (float)event->y());
		mouseEvent.buttonType = mappingMouseButton(event);
		mouseEvent.actionType = PActionType::AT_PRESS;
		mouseEvent.mods = mappingModifierBits(event->modifiers());
		mouseEvent.camera = camera;
		mouseEvent.x = (float)event->x();
		mouseEvent.y = (float)event->y();

		auto activeScene = SceneGraphFactory::instance()->active();

		activeScene->onMouseEvent(mouseEvent);

		mImWindow.mousePressEvent(mouseEvent);

		updateGrpahicsContext();
	}

	void POpenGLWidget::mouseReleaseEvent(QMouseEvent *event)
	{
		// do picking
		if( (event->modifiers() == 0 || event->modifiers() == Qt::ShiftModifier)
			&& event->button() == Qt::LeftButton 
			&& !ImGuizmo::IsUsing()
			&& !ImGui::GetIO().WantCaptureMouse)
		{
			int x = event->x();
			int y = event->y();

			int w = std::abs(mCursorX - x);
			int h = std::abs(mCursorY - y);
			x = std::min(mCursorX, x);
			y = std::min(mCursorY, y);
			// flip y
			y = this->height() - y - 1;

			makeCurrent();
			const auto& selection = this->select(x, y, w, h);
			doneCurrent();
		}


		auto camera = this->getCamera();

		mButtonState = QButtonState::QBUTTON_UP;

		PMouseEvent mouseEvent;
		mouseEvent.ray = camera->castRayInWorldSpace((float)event->x(), (float)event->y());
		mouseEvent.buttonType = mappingMouseButton(event);
		mouseEvent.actionType = PActionType::AT_RELEASE;
		mouseEvent.mods = mappingModifierBits(event->modifiers());
		mouseEvent.camera = camera;
		mouseEvent.x = (float)event->x();
		mouseEvent.y = (float)event->y();

		auto activeScene = SceneGraphFactory::instance()->active();

		activeScene->onMouseEvent(mouseEvent);

		mImWindow.mouseReleaseEvent(mouseEvent);

		updateGrpahicsContext();
	}

	void POpenGLWidget::mouseMoveEvent(QMouseEvent *event)
	{
		auto camera = this->getCamera();

		if (event->buttons().testFlag(Qt::LeftButton) &&
			mButtonState == QBUTTON_DOWN &&
			event->modifiers() == Qt::AltModifier &&
			!mImWindow.cameraLocked())
		{
			camera->rotateToPoint(event->x(), event->y());
		}
		else if (event->buttons().testFlag(Qt::RightButton) &&
			mButtonState == QBUTTON_DOWN &&
			event->modifiers() == Qt::AltModifier &&
			!mImWindow.cameraLocked())
		{
			camera->translateToPoint(event->x(), event->y());
		}

		PMouseEvent mouseEvent;
		mouseEvent.ray = camera->castRayInWorldSpace((float)event->x(), (float)event->y());
		mouseEvent.buttonType = mappingMouseButton(event);
		mouseEvent.actionType = PActionType::AT_REPEAT;
		mouseEvent.mods = mappingModifierBits(event->modifiers());
		mouseEvent.camera = camera;
		mouseEvent.x = (float)event->x();
		mouseEvent.y = (float)event->y();

		auto activeScene = SceneGraphFactory::instance()->active();

		activeScene->onMouseEvent(mouseEvent);

		mImWindow.mouseMoveEvent(mouseEvent);

		updateGrpahicsContext();
	}

	void POpenGLWidget::wheelEvent(QWheelEvent *event)
	{
		if(!mImWindow.cameraLocked())
			this->getCamera()->zoom(-0.001*event->angleDelta().x());

		update();
	}

	void POpenGLWidget::keyPressEvent(QKeyEvent* event)
	{
		auto activeScene = SceneGraphFactory::instance()->active();

		PKeyboardEvent keyEvent;
		keyEvent.key = KeyMap[event->key()];
		keyEvent.action = AT_PRESS;
		keyEvent.mods = mappingModifierBits(event->modifiers());

		activeScene->onKeyboardEvent(keyEvent);
	}

	void POpenGLWidget::keyReleaseEvent(QKeyEvent* event)
	{
		auto activeScene = SceneGraphFactory::instance()->active();

		PKeyboardEvent keyEvent;
		keyEvent.key = KeyMap[event->key()];
		keyEvent.action = AT_RELEASE;
		keyEvent.mods = mappingModifierBits(event->modifiers());

		activeScene->onKeyboardEvent(keyEvent);
	}

	void POpenGLWidget::onSelected(const Selection& s)
	{
		if (s.items.empty())
			return;

		emit this->nodeSelected(s.items[0].node);

	}

	void POpenGLWidget::updateGrpahicsContext()
	{
		makeCurrent();

		PSimulationThread::instance()->startUpdatingGraphicsContext();

		SceneGraphFactory::instance()->active()->updateGraphicsContext();

		PSimulationThread::instance()->stopUpdatingGraphicsContext();

		update();

		doneCurrent();
	}

	void POpenGLWidget::updateGraphicsContext(Node* node)
	{
		makeCurrent();

		PSimulationThread::instance()->startUpdatingGraphicsContext();

		node->graphicsPipeline()->forceUpdate();

		PSimulationThread::instance()->stopUpdatingGraphicsContext();

		update();

		doneCurrent();
	}

}