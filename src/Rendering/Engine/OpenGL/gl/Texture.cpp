#include "Texture.h"

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

namespace gl 
{
	void Texture::create()
	{
		glGenTextures(1, &id);
	}

	void Texture::release()
	{
		glDeleteTextures(1, &id);
	}

	void Texture::bind()
	{
		glBindTexture(target, id);
	}

	void Texture::bind(int slot)
	{
		glActiveTexture(slot);
		glBindTexture(target, id);
	}

	void Texture::unbind()
	{
		glBindTexture(target, 0);
	}


	Texture2D::Texture2D()
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

	void Texture2D::create()
	{
		Texture::create();
		glBindTexture(target, id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, maxFilter);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, glm::value_ptr(borderColor));

		glCheckError();
	}

	void Texture2D::resize(int w, int h)
	{
		glBindTexture(target, id);
		glTexImage2D(target, 0, internalFormat, w, h, 0, format, type, 0);
		glCheckError();
	}


	void Texture2D::load(int w, int h, void* data)
	{
		glBindTexture(target, id);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(target, 0, internalFormat, w, h, 0, format, type, data);
		glCheckError();
	}

	void Texture2D::dump(void* pixels)
	{
		glBindTexture(target, id);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glGetTexImage(target, 0, internalFormat, type, pixels);
		glCheckError();
	}


	void Texture2D::clear(void* value)
	{
		glClearTexImage(id, 0, format, type, value);
		glCheckError();
	}
	void Texture2D::genMipmap()
	{
		glBindTexture(target, id);
		glGenerateMipmap(target);
		glCheckError();
	}


	Texture2DArray::Texture2DArray()
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

	void Texture2DArray::create()
	{
		Texture::create();
		glBindTexture(target, id);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, minFilter);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, maxFilter);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, wrapS);
		glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, wrapT);
		glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, glm::value_ptr(borderColor));
		glCheckError();
	}

	void Texture2DArray::resize(int w, int h, int layers)
	{
		glBindTexture(target, id);
		glTexImage3D(target, 0, internalFormat, w, h, layers, 0, format, type, 0);
		glCheckError();
	}
}

