#pragma once

#include <Wt/WContainerWidget.h>

#include <memory>

#include <GraphicsObject/Framebuffer.h>
#include <GraphicsObject/Texture.h>

#include <ImWidgets/ImWindow.h>
#include <RenderWindow.h>

namespace dyno
{
	class Camera;
	class SceneGraph;
	class GLRenderEngine;

	struct RenderParams;
};

struct GLFWwindow;
class ImageEncoder;
class ImGuiBackendWt;

class WSimulationCanvas
	: public Wt::WContainerWidget
	, public dyno::RenderWindow
{
public:
	WSimulationCanvas();
	~WSimulationCanvas();

	void setScene(std::shared_ptr<dyno::SceneGraph> scene);

	void update();

public:
	//Mouse interaction
	void onMousePressed(const Wt::WMouseEvent& evt);
	void onMouseDrag(const Wt::WMouseEvent& evt);
	void onMouseReleased(const Wt::WMouseEvent& evt);
	void onMouseWheeled(const Wt::WMouseEvent& evt);

	//Keyboard interaction
	void onKeyWentDown(const Wt::WKeyEvent& evt);
	void onKeyWentUp(const Wt::WKeyEvent& evt);

protected:
	void initializeGL();
	void makeCurrent();
	void doneCurrent();

	void render(Wt::WFlags<Wt::RenderFlag> flags) override;
	void layoutSizeChanged(int width, int height) override;

	int width = 800;
	int height = 600;

private:
	Wt::WImage* mImage;
	Wt::WApplication* mApp;

	GLFWwindow* mContext;
	ImGuiBackendWt* mImGuiCtx;

	dyno::ImWindow mImWindow;

	std::vector<unsigned char> mImageData;					// raw image	
	std::vector<unsigned char> mJpegBuffer;					// jpeg data	
	std::unique_ptr<ImageEncoder> mJpegEncoder;				// jpeg encoder	
	std::unique_ptr<Wt::WMemoryResource> mJpegResource;		// Wt resource for jpeg image

	std::shared_ptr<dyno::SceneGraph> mScene = nullptr;
	//std::shared_ptr<dyno::Camera>	  mCamera;

	// internal framebuffer
	dyno::Framebuffer mFramebuffer;
	dyno::Texture2D	mFrameColor;

	bool mMouseButtonDown = false;

	int mCursorX = -1;
	int mCursorY = -1;
	int mtempCursorX = -1;
};