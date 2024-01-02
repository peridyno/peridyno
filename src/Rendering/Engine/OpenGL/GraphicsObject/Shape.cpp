#include "Shape.h"

#include "glad/glad.h"

namespace dyno
{
	Shape::Shape()
	{
	}

	Shape::~Shape()
	{
	}

	void Shape::create()
	{
		if (!mInitialized)
		{
			glVertexIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			glNormalIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			glTexCoordIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

			mInitialized = true;
		}
	}

	void Shape::release()
	{
		glVertexIndex.release();
		glNormalIndex.release();
		glTexCoordIndex.release();
	}

	void Shape::update()
	{
		glVertexIndex.load(vertexIndex);
		glNormalIndex.load(normalIndex);
		glTexCoordIndex.load(texCoordIndex);
	}

	void Shape::updateGL()
	{
		glVertexIndex.updateGL();
		glNormalIndex.updateGL();
		glTexCoordIndex.updateGL();
	}

}