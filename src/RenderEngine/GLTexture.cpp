#include "GLTexture.h"

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include "Utility.h"

namespace dyno 
{
	void GLTexture::create()
	{
		glGenTextures(1, &id);
	}

	void GLTexture::release()
	{
		glDeleteTextures(1, &id);
	}

	void GLTexture::bind()
	{
		glBindTexture(target, id);
	}

	void GLTexture::bind(int slot)
	{
		glActiveTexture(slot);
		glBindTexture(target, id);
	}

	void GLTexture::unbind()
	{
		glBindTexture(target, 0);
	}


	GLTexture2D::GLTexture2D()
	{
		this->target = GL_TEXTURE_2D;

		// default value...
		this->format = GL_RGBA;
		this->internalFormat = GL_RGBA32F;
		this->type = GL_FLOAT;

		this->minFilter = GL_NEAREST;
		this->maxFilter = GL_NEAREST;

		this->wrapS = GL_CLAMP_TO_BORDER;
		this->wrapT = GL_CLAMP_TO_BORDER;

		this->borderColor = glm::vec4(1);
	}

	void GLTexture2D::create()
	{
		GLTexture::create();
		glBindTexture(target, id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, maxFilter);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, glm::value_ptr(borderColor));

		glCheckError();
	}

	void GLTexture2D::resize(int w, int h)
	{
		glBindTexture(target, id);
		glTexImage2D(target, 0, internalFormat, w, h, 0, format, type, 0);
		glCheckError();
	}


	void GLTexture2D::load(int w, int h, void* data)
	{
		glBindTexture(target, id);
		glTexImage2D(target, 0, internalFormat, w, h, 0, format, type, data);
		glCheckError();
	}


	void GLTexture2D::clear(void* value)
	{
		glClearTexImage(id, 0, format, type, value);
		glCheckError();
	}
	void GLTexture2D::genMipmap()
	{
		glBindTexture(target, id);
		glGenerateMipmap(target);
		glCheckError();
	}


	GLTexture2DArray::GLTexture2DArray()
	{
		this->target = GL_TEXTURE_2D_ARRAY;

		// default value...
		this->format = GL_RGBA;
		this->internalFormat = GL_RGBA32F;
		this->type = GL_FLOAT;

		this->minFilter = GL_NEAREST;
		this->maxFilter = GL_NEAREST;

		this->wrapS = GL_CLAMP_TO_BORDER;
		this->wrapT = GL_CLAMP_TO_BORDER;

		this->borderColor = glm::vec4(1);
	}

	void GLTexture2DArray::create()
	{
		GLTexture::create();
		glBindTexture(target, id);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, minFilter);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, maxFilter);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, wrapS);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, wrapT);
		glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, glm::value_ptr(borderColor));
		glCheckError();
	}

	void GLTexture2DArray::resize(int w, int h, int layers)
	{
		glBindTexture(target, id);
		glTexImage3D(target, 0, internalFormat, w, h, layers, 0, format, type, 0);
		glCheckError();
	}
}

