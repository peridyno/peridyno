#include "PointRender.h"
// opengl
#include <glad/glad.h>
#include "RenderEngine.h"
#include "Utility.h"

#include <cuda_gl_interop.h>

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
	
	glCheckError();

	return true;
}


void PointRenderer::updateGL()
{
// 	Node* parent = getParent();
// 
// 	if (parent == NULL)
// 	{
// 		//SPDLOG_ERROR("Should insert this module into a node!");
// 		return;
// 	}
// 
// 	if (!parent->isVisible())
// 		return;
// 
// 	ParticleSystem<DataType3f>* pParticleSystem = TypeInfo::cast<ParticleSystem<DataType3f>>(parent);
// 
// 	if (pParticleSystem != nullptr)
// 	{
// 		// first try to solve particle system
// 		auto force = pParticleSystem->currentForce()->getReference();
// 		auto position = pParticleSystem->currentPosition()->getReference();
// 		auto velocity = pParticleSystem->currentVelocity()->getReference();
// 
// 		mNumPoints = position->size();
// 		mPosition.loadCuda(position->getDataPtr(), mNumPoints * sizeof(float) * 3);
// 		mVelocity.loadCuda(velocity->getDataPtr(), mNumPoints * sizeof(float) * 3);
// 		mForce.loadCuda(force->getDataPtr(), mNumPoints * sizeof(float) * 3);
// 	}
// 	else
// 	{
// 		auto pPointSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(parent->getTopologyModule());
// 		if (pPointSet == nullptr)
// 		{
// 			//SPDLOG_ERROR("PointRenderModule: The topology module is not supported!");
// 			return;
// 		}
// 		if (!pPointSet->isInitialized())
// 		{
// 			pPointSet->initialize();
// 		}
// 		auto& xyz = pPointSet->getPoints();
// 		mNumPoints = xyz.size();
// 		mPosition.loadCuda(xyz.getDataPtr(), mNumPoints * sizeof(float) * 3);
// 	}
}

void PointRenderer::paintGL()
{
	mVertexArray.bind();
	glDrawArrays(GL_POINTS, 0, mNumPoints);
	glCheckError();
}