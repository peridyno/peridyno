#include "OpenGLContext.h"

namespace dyno
{
	OpenGLContext& OpenGLContext::getInstance()
	{
		static OpenGLContext m_openglContext;

		return m_openglContext;
	}

	bool OpenGLContext::initialize()
	{
		if (m_initialized)
		{
			return true;
		}

// 		if (glewInit() != GLEW_OK)
// 		{
// 			return false;
// 		}

		m_initialized = true;
		return m_initialized;
	}
}


