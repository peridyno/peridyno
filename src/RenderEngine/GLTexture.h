#pragma once

#include "GLObject.h"
#include <glm/glm.hpp>

class GLTexture : public GLObject
{
public:
	virtual void create();
	virtual void release();

	virtual void bind();
	virtual void unbind();

	virtual void bind(int slot);

public:
	unsigned int target;
	unsigned int internalFormat;
	unsigned int format;
	unsigned int type;

	unsigned int minFilter;
	unsigned int maxFilter;
	unsigned int wrapS;
	unsigned int wrapT;

	glm::vec4	 borderColor;
};

class GLTexture2D : public GLTexture
{
public:
	GLTexture2D();

	virtual void create();

	virtual void resize(int w, int h);
	virtual void load(int w, int h, void* data);

	// OpenGL 4.4+, clear texture
	virtual void clear(void* value);
};