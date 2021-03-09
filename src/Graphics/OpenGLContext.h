#pragma once

#include <GL/glew.h>
#include <memory>

namespace dyno{

class OpenGLContext
{
public:
	static OpenGLContext& getInstance();

	bool initialize();
	bool isInitialized() { return m_initialized; }

private:
	explicit OpenGLContext() { m_initialized = false; }
	OpenGLContext(const OpenGLContext&) {};
	OpenGLContext& operator=(const OpenGLContext&) {};

	~OpenGLContext() {};


	
	bool m_initialized;
};

}