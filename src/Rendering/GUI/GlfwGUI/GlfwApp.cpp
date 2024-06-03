#include "GlfwApp.h"

namespace dyno 
{
	GlfwApp::GlfwApp(int argc /*= 0*/, char **argv /*= NULL*/)
	{
		//A hack to address the slow launching problem
#ifdef CUDA_BACKEND
		auto status = cudaSetDevice(0);
		if (status != cudaSuccess) {
			fprintf(stderr, "CUDA initialization failed!  Do you have a CUDA-capable GPU installed?");
			exit(0);
		}
		cudaFree(0);
#endif // CUDA_BACKEND
	}

	GlfwApp::~GlfwApp()
	{
	}

	void GlfwApp::initialize(int width, int height, bool usePlugin)
	{
		mRenderWindow = std::make_shared<GlfwRenderWindow>();

		mRenderWindow->initialize(width, height);

		//Load envmap
		mRenderWindow->getRenderEngine()->setDefaultEnvmap();
	}

	void GlfwApp::mainLoop()
	{
		mRenderWindow->mainLoop();
	}

	void GlfwApp::setWindowTitle(const std::string& title)
	{
		mRenderWindow->setWindowTitle(title);
	}

}