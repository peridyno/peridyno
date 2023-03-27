#include "Texture.h"

#include <glad/glad.h>

#include <iostream>

namespace gl 
{

	Texture::Texture()
	{
		// default value...
		this->format = GL_RGBA;
		this->internalFormat = GL_RGBA32F;
		this->type = GL_FLOAT;

		this->minFilter = GL_LINEAR;	//GL_LINEAR_MIPMAP_LINEAR
		this->maxFilter = GL_LINEAR;
	}


	void Texture::create()
	{
		if (target == -1) {
			std::cerr << "Failed to create texture, wrong target id: " << target << std::endl;
			return;
		}

		glGenTextures(1, &id);

		glBindTexture(target, id);
		glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter);
		glTexParameteri(target, GL_TEXTURE_MAG_FILTER, maxFilter);
		glCheckError();
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
		glCheckError();
	}

	void Texture::dump(void* pixels)
	{
		glBindTexture(target, id);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glGetTexImage(target, 0, internalFormat, type, pixels);
		glCheckError();
	}

	Texture2D::Texture2D()
	{
		this->target = GL_TEXTURE_2D;
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
	}


	void Texture2DArray::resize(int w, int h, int layers)
	{
		glBindTexture(target, id);
		glTexImage3D(target, 0, internalFormat, w, h, layers, 0, format, type, 0);
		glCheckError();
	}
}

