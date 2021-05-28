#include "FluidRender.h"

// opengl
#include <glad/glad.h>
#include "RenderEngine.h"
#include "Utility.h"


using namespace dyno;

IMPLEMENT_CLASS_COMMON(FluidRenderer, 0)

FluidRenderer::FluidRenderer()
{
	mPointSize = 0.005f;
	mNumPoints = 0;

	this->setName("fluid_renderer");

}

void FluidRenderer::setPointSize(float size)
{
	mPointSize = size;
}

float FluidRenderer::getPointSize() const
{
	return mPointSize;
}

bool FluidRenderer::initializeGL()
{
	mPointBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

	mVertexArray.create();
	mVertexArray.bindVertexBuffer(&mPointBuffer, 0, 3, GL_FLOAT, 0, 0, 0);

	glCheckError();
	return true;
}

void FluidRenderer::updateGL()
{
// 	using namespace dyno;
// 	Node* parent = getParent();
// 
// 	if (parent == NULL || !parent->isVisible())
// 		return;
// 
// 	ParticleSystem<DataType3f>* pParticleSystem = TypeInfo::CastPointerUp<ParticleSystem<DataType3f>>(parent);
// 
// 	if (pParticleSystem != nullptr)
// 	{
// 		auto position = pParticleSystem->currentPosition()->getReference();
// 
// 		mNumPoints = position->size();
// 		mPointBuffer.loadCuda(position->getDataPtr(), mNumPoints * sizeof(float) * 3);
// 	}
// 	else
// 	{
// 		//SPDLOG_WARN("FluidRenderer: failed to get parent ParticleSystem");
// 	}
}

void FluidRenderer::paintGL()
{
	mVertexArray.bind();
	glDrawArrays(GL_POINTS, 0, mNumPoints);
	glCheckError();
}