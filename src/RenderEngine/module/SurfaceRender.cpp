#include "SurfaceRender.h"

// opengl
#include <glad/glad.h>
#include "RenderEngine.h"
#include "Utility.h"

// framework
#include "Topology/TriangleSet.h"
#include "Framework/Node.h"

using namespace dyno;

IMPLEMENT_CLASS_COMMON(SurfaceRenderer, 0)

SurfaceRenderer::SurfaceRenderer()
{
	this->setName("surface_renderer");
}

bool SurfaceRenderer::initializeGL()
{
	// create vertex buffer and vertex array object
	mVAO.create();
	mIndexBuffer.create(GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
	mVertexBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
	//mNormalBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

	mVAO.bindIndexBuffer(&mIndexBuffer);
	mVAO.bindVertexBuffer(&mVertexBuffer, 0, 3, GL_FLOAT, 0, 0, 0);
	//mVAO.bindVertexBuffer(&mNormalBuffer, 1, 3, GL_FLOAT);

	return true;
}

void SurfaceRenderer::updateGL()
{
	dyno::Node* parent = getParent();

	if (parent == NULL)
	{
		//SPDLOG_ERROR("Should insert this module into a node!");
		return;
	}

	if (!parent->isVisible())
		return;

	// dump geometry
	auto triSet = std::dynamic_pointer_cast<dyno::TriangleSet<dyno::DataType3f>>(parent->getTopologyModule());

	if (triSet == nullptr)
	{
		//SPDLOG_ERROR("Cannot get triangle mesh!");
		return;
	}

	auto triangles = triSet->getTriangles();
	auto vertices = triSet->getPoints();
	//auto normals = triSet->getNormals();
	
	mDrawCount = triangles->size() * 3;
	
	mVertexBuffer.loadCuda(vertices.begin(), vertices.size() * sizeof(float) * 3);
	//mNormalBuffer.loadCuda(normals.getDataPtr(), normals.size() * sizeof(float) * 3);
	mIndexBuffer.loadCuda(triangles->begin(), triangles->size() * sizeof(unsigned int) * 3);

}

void SurfaceRenderer::paintGL()
{
	mVAO.bind();
	glDrawElements(GL_TRIANGLES, mDrawCount, GL_UNSIGNED_INT, 0);
	mVAO.unbind();

	glCheckError();
}