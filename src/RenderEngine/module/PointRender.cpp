#include "PointRender.h"
// opengl
#include <glad/glad.h>
#include "RenderEngine.h"
#include "Utility.h"

#include <cuda_gl_interop.h>

// framework
#include "Topology/TriangleSet.h"
#include "Framework/Node.h"

using namespace dyno;

IMPLEMENT_CLASS_COMMON(PointRenderer, 0)

PointRenderer::PointRenderer()
{
	mPointSize = 0.001f;
	mNumPoints = 1;
	this->setName("point_renderer");
}

void PointRenderer::setPointSize(float size)
{
	mPointSize = size;
}

float PointRenderer::getPointSize() const
{
	return mPointSize;
}


bool PointRenderer::initializeGL()
{	
	mPosition.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
	mVelocity.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
	mForce.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

	mVertexArray.create();
	mVertexArray.bindVertexBuffer(&mPosition, 0, 3, GL_FLOAT, 0, 0, 0);
	mVertexArray.bindVertexBuffer(&mVelocity, 1, 3, GL_FLOAT, 0, 0, 0);
	mVertexArray.bindVertexBuffer(&mForce, 2, 3, GL_FLOAT, 0, 0, 0);
	
	mShaderProgram = CreateShaderProgram("point.vert", "point.frag");
	
	glCheckError();

	return true;
}


void PointRenderer::updateGL()
{
 	Node* parent = getParent();
 
 	if (parent == NULL)
 	{
 		return;
 	}
 
 	if (!parent->isVisible())
 		return;

 	auto pPointSet = TypeInfo::cast<PointSet<DataType3f>>(parent->getTopologyModule());
 	if (pPointSet == nullptr)
 	{
 		return;
 	}
 	if (!pPointSet->isInitialized())
 	{
 		pPointSet->initialize();
 	}
 	auto& xyz = pPointSet->getPoints();
 	mNumPoints = xyz.size();
 	mPosition.loadCuda(xyz.begin(), mNumPoints * sizeof(float) * 3);

}

void PointRenderer::paintGL()
{
	unsigned int subroutine = 0;
	mShaderProgram.use();
	mShaderProgram.setFloat("uPointSize", this->getPointSize());
	glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);

	mVertexArray.bind();
	glDrawArrays(GL_POINTS, 0, mNumPoints);
	glCheckError();
}