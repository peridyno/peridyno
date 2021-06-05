#include "VtkPointVisualModule.h"
// opengl
#include <glad/glad.h>
#include "RenderEngine.h"
// framework
#include "Topology/TriangleSet.h"
#include "Framework/Node.h"

#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkPointSource.h>
#include <vtkRenderer.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOpenGLVertexBufferObjectCache.h>
#include <vtkOpenGLVertexBufferObjectGroup.h>
#include <vtkPolyData.h>
#include <vtkFloatArray.h>
#include <vtkOpenGLVertexBufferObject.h>
#include <vtkOpenGLIndexBufferObject.h>

#include <cuda_gl_interop.h>

using namespace dyno;

class PointMapper : public vtkOpenGLPolyDataMapper
{
public:
	PointMapper(PointVisualModule* v): m_module(v)
	{
		// create psedo data, required by the vtkOpenGLPolyDataMapper to render content
		vtkNew<vtkPointSource> psedoData;
		psedoData->SetCenter(0, 0, 0);
		psedoData->SetNumberOfPoints(10);
		psedoData->SetRadius(1.0);
		SetInputConnection(psedoData->GetOutputPort());
	}

	void ComputeBounds() override
	{
		// TODO: we might need the accurate bound of the node
		this->Bounds[0] = 0;
		this->Bounds[1] = 1;
		this->Bounds[2] = 0;
		this->Bounds[3] = 1;
		this->Bounds[4] = 0;
		this->Bounds[5] = 1;
	}

	void UpdateBufferObjects(vtkRenderer *ren, vtkActor *act) override
	{
		// TODO: we need some mechanism to check whether the VBO need update

		if (!m_module->isInitialized())	return;

		auto node = m_module->getParent();

		if (node == NULL || !node->isVisible())	return;
		
		auto pSet = std::dynamic_pointer_cast<dyno::PointSet<dyno::DataType3f>>(node->getTopologyModule());
		auto verts = pSet->getPoints();

		cudaError_t error;

		if (!m_initialized)
		{
			printf("Intialize\n");
			m_initialized = true;
								
			// vertex buffer
			vtkNew<vtkPoints>    tempVertData;
			tempVertData->SetNumberOfPoints(verts.size());

			vtkOpenGLRenderWindow* renWin = vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow());
			vtkOpenGLVertexBufferObjectCache* cache = renWin->GetVBOCache();
			this->VBOs->CacheDataArray("vertexMC", tempVertData->GetData(), cache, VTK_FLOAT);
			this->VBOs->BuildAllVBOs(cache);
			vtkOpenGLVertexBufferObject* vertexBuffer = this->VBOs->GetVBO("vertexMC");
						
			// index buffer
			// this->Primitives[PrimitivePoints].IBO;
			std::vector<unsigned int> indexArray(verts.size());
			for (unsigned int i = 0; i < indexArray.size(); i++)
				indexArray[i] = i;
			
			this->Primitives[PrimitivePoints].IBO->Upload(indexArray, vtkOpenGLIndexBufferObject::ElementArrayBuffer);
			this->Primitives[PrimitivePoints].IBO->IndexCount = indexArray.size();

			// create memory mapper for CUDA
			error = cudaGraphicsGLRegisterBuffer(&m_cudaVBO, vertexBuffer->GetHandle(), cudaGraphicsRegisterFlagsWriteDiscard);
		}

		// copy vertex memory
		{
			size_t size;
			void*  cudaPtr = 0;

			// upload vertex
			error = cudaGraphicsMapResources(1, &m_cudaVBO);
			error = cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, m_cudaVBO);
			error = cudaMemcpy(cudaPtr, verts.begin(), verts.size() * sizeof(float) * 3, cudaMemcpyDeviceToDevice);
			error = cudaGraphicsUnmapResources(1, &m_cudaVBO);
		}
	}

private:
	dyno::PointVisualModule* m_module;

	bool				m_initialized = false;

	cudaGraphicsResource*			m_cudaVBO;
};

IMPLEMENT_CLASS_COMMON(PointVisualModule, 0)

PointVisualModule::PointVisualModule()
{
	this->setName("point_renderer");
	createActor();
}

void PointVisualModule::createActor()
{
	m_actor = vtkActor::New();
	m_actor->GetProperty()->SetRepresentationToPoints(); 
	m_actor->GetProperty()->RenderPointsAsSpheresOn();
	m_actor->GetProperty()->SetPointSize(2.0);
	this->m_mapper = new PointMapper(this);
	m_actor->SetMapper(m_mapper);	
}

void PointVisualModule::updateRenderingContext()
{
	// TODO: update VBO here?
}