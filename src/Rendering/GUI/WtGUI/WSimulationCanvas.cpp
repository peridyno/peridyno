#include "WSimulationCanvas.h"
#include "ImageEncoder.h"

#include <Wt/WApplication.h>
#include <Wt/WMemoryResource.h>
#include <Wt/WImage.h>

#include <GLFW/glfw3.h>

#include <SceneGraph.h>
#include <GLRenderEngine.h>
#include <OrbitCamera.h>
#include <TrackballCamera.h>

#include "imgui_impl_wt.h"
#include "ImWidget.h"
#include <ImGuizmo.h>

using namespace dyno;

std::map<Wt::Key, PKeyboardType> WKeyMap =
{
	{Wt::Key::Unknown, PKEY_UNKNOWN},
	{Wt::Key::Space, PKEY_SPACE},
	{Wt::Key::Key_0, PKEY_0},
	{Wt::Key::Key_1, PKEY_1},
	{Wt::Key::Key_2, PKEY_2},
	{Wt::Key::Key_3, PKEY_3},
	{Wt::Key::Key_4, PKEY_4},
	{Wt::Key::Key_5, PKEY_5},
	{Wt::Key::Key_6, PKEY_6},
	{Wt::Key::Key_7, PKEY_7},
	{Wt::Key::Key_8, PKEY_8},
	{Wt::Key::Key_9, PKEY_9},
	{Wt::Key::A, PKEY_A},
	{Wt::Key::B, PKEY_B},
	{Wt::Key::C, PKEY_C},
	{Wt::Key::D, PKEY_D},
	{Wt::Key::E, PKEY_E},
	{Wt::Key::F, PKEY_F},
	{Wt::Key::G, PKEY_G},
	{Wt::Key::H, PKEY_H},
	{Wt::Key::I, PKEY_I},
	{Wt::Key::J, PKEY_J},
	{Wt::Key::K, PKEY_K},
	{Wt::Key::L, PKEY_L},
	{Wt::Key::M, PKEY_M},
	{Wt::Key::N, PKEY_N},
	{Wt::Key::O, PKEY_O},
	{Wt::Key::P, PKEY_P},
	{Wt::Key::Q, PKEY_Q},
	{Wt::Key::R, PKEY_R},
	{Wt::Key::S, PKEY_S},
	{Wt::Key::T, PKEY_T},
	{Wt::Key::U, PKEY_U},
	{Wt::Key::V, PKEY_V},
	{Wt::Key::W, PKEY_W},
	{Wt::Key::X, PKEY_X},
	{Wt::Key::Y, PKEY_Y},
	{Wt::Key::Z, PKEY_Z},
	{Wt::Key::Escape, PKEY_ESCAPE},
	{Wt::Key::Enter, PKEY_ENTER},
	{Wt::Key::Tab, PKEY_TAB},
	{Wt::Key::Backspace, PKEY_BACKSPACE},
	{Wt::Key::Insert, PKEY_INSERT},
	{Wt::Key::Delete, PKEY_DELETE},
	{Wt::Key::Right, PKEY_RIGHT},
	{Wt::Key::Left, PKEY_LEFT},
	{Wt::Key::Down, PKEY_DOWN},
	{Wt::Key::Up, PKEY_UP},
	{Wt::Key::PageUp, PKEY_PAGE_UP},
	{Wt::Key::PageDown, PKEY_PAGE_DOWN},
	{Wt::Key::Home, PKEY_HOME},
	{Wt::Key::End, PKEY_END},
	{Wt::Key::F1, PKEY_F1},
	{Wt::Key::F2, PKEY_F2},
	{Wt::Key::F3, PKEY_F3},
	{Wt::Key::F4, PKEY_F4},
	{Wt::Key::F5, PKEY_F5},
	{Wt::Key::F6, PKEY_F6},
	{Wt::Key::F7, PKEY_F7},
	{Wt::Key::F8, PKEY_F8},
	{Wt::Key::F9, PKEY_F9},
	{Wt::Key::F10, PKEY_F10},
	{Wt::Key::F11, PKEY_F11},
	{Wt::Key::F12, PKEY_F12}
};

PModifierBits mappingWtModifierBits(Wt::KeyboardModifier mods)
{
	if (mods == Wt::KeyboardModifier::Control)
	{
		return PModifierBits::MB_CONTROL;
	}
	else if (mods == Wt::KeyboardModifier::Shift)
	{
		return PModifierBits::MB_SHIFT;
	}
	else if (mods == Wt::KeyboardModifier::Alt)
	{
		return PModifierBits::MB_ALT;
	}
	else
		return PModifierBits::MB_NO_MODIFIER;
}

WSimulationCanvas::WSimulationCanvas()
{
	this->setLayoutSizeAware(true);
	this->setStyleClass("remote-framebuffer");
	this->resize("100%", "90%");

	this->mouseWentUp().preventDefaultAction(true);
	this->mouseWentDown().preventDefaultAction(true);
	this->mouseDragged().preventDefaultAction(true);
	this->touchStarted().preventDefaultAction(true);
	this->touchMoved().preventDefaultAction(true);
	this->touchEnded().preventDefaultAction(true);

	this->mouseWentDown().connect(this, &WSimulationCanvas::onMousePressed);
	this->mouseWheel().connect(this, &WSimulationCanvas::onMouseWheeled);
	this->mouseDragged().connect(this, &WSimulationCanvas::onMouseDrag);
	this->mouseWentUp().connect(this, &WSimulationCanvas::onMouseReleased);

	//this->keyWentDown().connect(this, &WSimulationCanvas::onKeyWentDown);
	//this->keyWentUp().connect(this, &WSimulationCanvas::onKeyWentUp);

	this->setAttributeValue("oncontextmenu", "return false;");

	mApp = Wt::WApplication::instance();

	mImage = this->addNew<Wt::WImage>();
	mImage->resize("100%", "100%");

	mImage->setJavaScriptMember("currURL", "null");
	mImage->setJavaScriptMember("nextURL", "null");
	mImage->setJavaScriptMember("onload",
		"function() {"
		"this.currURL = this.nextURL;"
		"this.nextURL = null;"
		"if (this.currURL != null) {"
		"this.src = this.currURL;"
		"}"
		"}.bind(" + mImage->jsRef() + ")");
	mImage->setJavaScriptMember("onerror",
		"function() {"
		"this.currURL = this.nextURL;"
		"this.nextURL = null;"
		"if (this.currURL != null) {"
		"this.src = this.currURL;"
		"}"
		"}.bind(" + mImage->jsRef() + ")");
	mImage->setJavaScriptMember("update",
		"function(url) { "
		"if (this.currURL == null) {"
		"this.currURL = url;"
		"this.src = this.currURL;"
		"} else {"						// still loading
		"this.nextURL = url;"
		"}"
		"}.bind(" + mImage->jsRef() + ")");

	mImageData.resize(width * height * 3);	// initialize image buffer
	mJpegEncoder = std::make_unique<ImageEncoderNV>();
	mJpegEncoder->SetQuality(100);
	mJpegResource = std::make_unique<Wt::WMemoryResource>("image/jpeg");

	mRenderEngine = std::make_shared<dyno::GLRenderEngine>();

	this->setWindowSize(width, height);

	// initialize OpenGL context and RenderEngine
	this->initializeGL();

	//this->toggleImGUI();
}

WSimulationCanvas::~WSimulationCanvas()
{
	makeCurrent();

	delete mImGuiCtx;

	mRenderEngine->terminate();

	mFramebuffer.release();
	mFrameColor.release();

	mImageData.resize(0);
	mJpegBuffer.resize(0);

	glfwDestroyWindow(mContext);
	//glfwTerminate();
	Wt::log("warning") << "WSimulationCanvas destory";
}

void WSimulationCanvas::initializeGL()
{
	// initialize render engine and target
	glfwInit();
	// Set all the required options for GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	mContext = glfwCreateWindow(width, height, "", NULL, NULL);
	if (!mContext)
	{
		Wt::log("error") << "Failed to create OpenGL context!";
		exit(-1);
	}

	makeCurrent();

	mRenderEngine->initialize();

	if (showImGUI())
	{
		mImGuiCtx = new ImGuiBackendWt(this);
		// Setup Dear ImGui context
		ImGui::StyleColorsDark();

		// Get Context scale
		float xscale, yscale;
		glfwGetMonitorContentScale(glfwGetPrimaryMonitor(), &xscale, &yscale);

		// Initialize ImWindow
		mImWindow.initialize(xscale);
		mImWindow.setEnableViewManipulate(false);
	}

	// create framebuffer here...
	mFrameColor.format = GL_RGB;
	mFrameColor.internalFormat = GL_RGB;
	mFrameColor.type = GL_BYTE;
	mFrameColor.create();
	mFrameColor.resize(width, height);

	mFramebuffer.create();
	mFramebuffer.bind();
	const unsigned int GL_COLOR_ATTACHMENT0 = 0x8CE0;
	mFramebuffer.setTexture(GL_COLOR_ATTACHMENT0, &mFrameColor);	//
	unsigned int buffers[]{ GL_COLOR_ATTACHMENT0 };
	mFramebuffer.drawBuffers(1, buffers);
	mFramebuffer.unbind();

	doneCurrent();
}

void WSimulationCanvas::makeCurrent()
{
	if (glfwGetCurrentContext() != mContext)
		glfwMakeContextCurrent(mContext);
}

void WSimulationCanvas::doneCurrent()
{
	glfwMakeContextCurrent(NULL);
}

void WSimulationCanvas::layoutSizeChanged(int width, int height)
{
	mCamera->setWidth(width);
	mCamera->setHeight(height);
	mImageData.resize(width * height * 3);

	this->makeCurrent();
	// resize framebuffer
	mFrameColor.resize(width, height);
	this->doneCurrent();

	WContainerWidget::layoutSizeChanged(width, height);
	scheduleRender();
}

void WSimulationCanvas::onMousePressed(const Wt::WMouseEvent& evt)
{
	if (!mImGuiCtx->handleMousePressed(evt))
	{
		mMouseButtonDown = true;

		if (evt.button() == Wt::MouseButton::Right)
		{
			mtempCursorX = evt.widget().x;
		}

		Wt::Coordinates coord = evt.widget();
		mCamera->registerPoint(coord.x, coord.y);
	}

	mCursorX = evt.widget().x;
	mCursorY = evt.widget().y;

	auto camera = this->getCamera();
	camera->registerPoint(evt.widget().x, evt.widget().y);

	scheduleRender();
}

void WSimulationCanvas::onMouseDrag(const Wt::WMouseEvent& evt)
{
	if (!mImGuiCtx->handleMouseDrag(evt))
	{
		auto mods = evt.modifiers();

		if (mods.test(Wt::KeyboardModifier::Alt) && mMouseButtonDown == true)
		{
			Wt::Coordinates coord = evt.widget();
			if (evt.button() == Wt::MouseButton::Left) {
				mCamera->rotateToPoint(coord.x, coord.y);
			}
			else if (evt.button() == Wt::MouseButton::Middle) {
				mCamera->translateToPoint(coord.x, coord.y);
			}
			else if (evt.button() == Wt::MouseButton::Right) {
				auto cam = this->getCamera();
				cam->zoom(-0.005 * float(evt.widget().x - mtempCursorX));
				mtempCursorX = evt.widget().x;
			}
		}
	}
	scheduleRender();
}

void WSimulationCanvas::onMouseReleased(const Wt::WMouseEvent& evt)
{
	if (!mImGuiCtx->handleMouseReleased(evt))
	{
		mMouseButtonDown = false;
	}

	if (this->getSelectionMode() == RenderWindow::OBJECT_MODE)
	{
		if (evt.button() == Wt::MouseButton::Left
			&& !ImGuizmo::IsUsing()
			&& !ImGui::GetIO().WantCaptureMouse)
		{
			int x = evt.widget().x;
			int y = evt.widget().y;
			int w = mCamera->viewportWidth();
			int h = mCamera->viewportHeight();

			makeCurrent();
			const auto& selection = this->select(x, y, w, h);
			doneCurrent();
		}
	}

	scheduleRender();
}

void WSimulationCanvas::onMouseWheeled(const Wt::WMouseEvent& evt)
{
	mCamera->zoom(-1.0 * evt.wheelDelta());
	scheduleRender();
}

void WSimulationCanvas::onKeyWentDown(const Wt::WKeyEvent& evt)
{
	PKeyboardEvent keyEvent;
	keyEvent.key = WKeyMap.find(evt.key()) == WKeyMap.end() ? PKEY_UNKNOWN : WKeyMap[evt.key()];
	keyEvent.action = AT_PRESS;
	keyEvent.mods = mappingWtModifierBits(evt.modifiers());

	mScene->onKeyboardEvent(keyEvent);

	switch (evt.key())
	{
	case Wt::Key::H:
		this->toggleImGUI();
		break;
	default:
		break;
	}

	scheduleRender();
}

void WSimulationCanvas::onKeyWentUp(const Wt::WKeyEvent& evt)
{
}

void WSimulationCanvas::render(Wt::WFlags<Wt::RenderFlag> flags)
{
	update();
}

void WSimulationCanvas::update()
{
	// do render
	{
		this->makeCurrent();

		if (mScene)
		{
			mScene->updateGraphicsContext();
		}

		// update rendering params
		mRenderParams.width = mCamera->viewportWidth();
		mRenderParams.height = mCamera->viewportHeight();
		mRenderParams.transforms.model = glm::mat4(1);	 // TODO: world transform?
		mRenderParams.transforms.view = mCamera->getViewMat();
		mRenderParams.transforms.proj = mCamera->getProjMat();
		mRenderParams.unitScale = mCamera->unitScale();

		mFramebuffer.bind();
		// heck: ImGUI widgets need to render twice...
		if (showImGUI())
		{
			// Start the Dear ImGui frame
			mImGuiCtx->NewFrame(mCamera->viewportWidth(), mCamera->viewportHeight());
			mImWindow.draw(this);
			mImGuiCtx->Render();
		}

		mRenderEngine->draw(mScene.get(), mRenderParams);

		if (showImGUI())
		{
			// Start the Dear ImGui frame
			mImGuiCtx->NewFrame(mCamera->viewportWidth(), mCamera->viewportHeight());
			mImWindow.draw(this);
			mImGuiCtx->Render();
		}

		// dump framebuffer
		mFrameColor.dump(mImageData.data());
		mFramebuffer.unbind();

		this->doneCurrent();
	}

	// encode image
	{
		mJpegBuffer.clear();

		mJpegEncoder->Encode(mImageData.data(),
			mCamera->viewportWidth(), mCamera->viewportHeight(), 0,
			mJpegBuffer);

		Wt::log("info") << mCamera->viewportWidth() << " x " << mCamera->viewportHeight()
			<< ", JPG size: " << mJpegBuffer.size() / 1024 << " kb";
	}

	// update UI
	{
		mJpegResource->setData(std::move(mJpegBuffer));
		const std::string url = mJpegResource->generateUrl();
		mImage->callJavaScriptMember("update", WWebWidget::jsStringLiteral(url));
	}
}

void WSimulationCanvas::setScene(std::shared_ptr<dyno::SceneGraph> scene)
{
	this->mScene = scene;

	// TODO: move to somewhere else!
	if (this->mScene)
	{
		makeCurrent();
		this->mScene->reset();
		doneCurrent();

		scheduleRender();
	}
}