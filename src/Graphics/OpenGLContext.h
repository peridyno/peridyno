#pragma once

#include <glad/gl.h>
#include <memory>

namespace dyno {
	class OpenGLContext
	{
	public:
		static OpenGLContext& getInstance();

		bool initialize();
		bool isInitialized() { return m_initialized; }

	private:
		explicit OpenGLContext() { m_initialized = false; }

		OpenGLContext(const OpenGLContext&) = delete;
		OpenGLContext& operator=(const OpenGLContext&) = delete;

		~OpenGLContext() {};

		bool m_initialized;
	};
}