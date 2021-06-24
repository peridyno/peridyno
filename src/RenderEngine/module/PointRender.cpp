#include "PointRender.h"
// opengl
#include <glad/glad.h>
#include "RenderEngine.h"
#include "RenderTools.h"
#include "Utility.h"

#include <cuda_gl_interop.h>

// framework
#include <Topology/TriangleSet.h>
#include <Framework/Node.h>

using namespace dyno;

IMPLEMENT_CLASS(PointRenderer)

PointRenderer::PointRenderer()
{
	mPointSize = 0.001f;
	mNumPoints = 1;
	this->setName("point_renderer");
}

PointRenderer::~PointRenderer()
{
	mColorBuffer.clear();
}

void PointRenderer::setPointSize(float size)
{
	mPointSize = size;
}

float PointRenderer::getPointSize() const
{
	return mPointSize;
}

void PointRenderer::setColorMapMode(ColorMapMode mode)
{
	mColorMode = mode;
}

void PointRenderer::setColorMapRange(float vmin, float vmax)
{
	mColorMin = vmin;
	mColorMax = vmax;
}

bool PointRenderer::initializeGL()
{	
	mPosition.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
	mColor.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

	mVertexArray.create();
	mVertexArray.bindVertexBuffer(&mPosition, 0, 3, GL_FLOAT, 0, 0, 0); 
	mVertexArray.bindVertexBuffer(&mColor, 1, 3, GL_FLOAT, 0, 0, 0);
	
	mShaderProgram = CreateShaderProgram("point.vert", "point.frag");
	
	glCheckError();

	return true;
}


void PointRenderer::updateGL()
{
 	Node* parent = getParent();
 
 	if (parent == NULL)
		return;
 
 	if (!parent->isVisible())
 		return;

	auto pPointSet = TypeInfo::cast<PointSet<DataType3f>>(parent->getTopologyModule());
	if (pPointSet == nullptr)
	{
		return;
	}

	auto& xyz = pPointSet->getPoints();
	mNumPoints = xyz.size();

	if (mColorBuffer.size() != mNumPoints) {
		mColorBuffer.resize(mNumPoints);
	}

	if (mColorMode == ColorMapMode::PER_VERTEX_SHADER && !this->inColor()->isEmpty())
	{
		mPosition.loadCuda(xyz.begin(), mNumPoints * sizeof(float) * 3);
		mColor.loadCuda(this->inColor()->getDataPtr()->begin(), mNumPoints * sizeof(float) * 3);
	}
	else
	{
		if (this->inColor()->isEmpty()) {
			RenderTools::setupColor(mColorBuffer, Vec3f(mBaseColor.r, mBaseColor.g, mBaseColor.b));
		}
		else {
			mColorBuffer.assign(this->inColor()->getData());
		}

		mPosition.loadCuda(xyz.begin(), mNumPoints * sizeof(float) * 3);
		mColor.loadCuda(mColorBuffer.begin(), mNumPoints * sizeof(float) * 3);
	}
}

void PointRenderer::paintGL(RenderMode mode)
{
	mShaderProgram.use();
	mShaderProgram.setFloat("uPointSize", this->getPointSize());

	unsigned int subroutine;
	if (mode == RenderMode::COLOR)
	{
		mShaderProgram.setVec3("uBaseColor", mBaseColor);
		mShaderProgram.setFloat("uMetallic", mMetallic);
		mShaderProgram.setFloat("uRoughness", mRoughness);
		mShaderProgram.setFloat("uAlpha", mAlpha);	// not implemented!
		
		mShaderProgram.setInt("uColorMode", mColorMode);
		mShaderProgram.setFloat("uColorMin", mColorMin);
		mShaderProgram.setFloat("uColorMax", mColorMax);

		subroutine = 0;
		glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
	}
	else if (mode == RenderMode::DEPTH)
	{
		subroutine = 1;
		glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
	}
	else
	{
		printf("Unknown render mode!\n");
		return;
	}

	mVertexArray.bind();
	glDrawArrays(GL_POINTS, 0, mNumPoints);
	glCheckError();
}
