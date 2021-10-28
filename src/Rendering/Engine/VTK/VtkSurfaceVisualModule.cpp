#include "VtkSurfaceVisualModule.h"

// framework
#include <Node.h>
#include <SceneGraph.h>
#include <Topology/TriangleSet.h>

#include <vtkActor.h>
#include <vtkCubeSource.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOpenGLRenderer.h>
#include <vtkOpenGLVertexBufferObjectCache.h>
#include <vtkOpenGLVertexBufferObjectGroup.h>
#include <vtkPolyData.h>
#include <vtkFloatArray.h>
#include <vtkOpenGLVertexBufferObject.h>
#include <vtkOpenGLIndexBufferObject.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <cuda_gl_interop.h>

using namespace dyno;

class SurfaceMapper : public vtkOpenGLPolyDataMapper
{
public:
	SurfaceMapper(VtkSurfaceVisualModule* v): m_module(v)
	{
		// create psedo data, required by the vtkOpenGLPolyDataMapper to render content
		vtkNew<vtkCubeSource> psedoData;
		SetInputConnection(psedoData->GetOutputPort());
	}

	void ComputeBounds() override
	{
		// TODO: we might need the accurate bound of the node
		this->GetInput()->GetBounds(this->Bounds);
	}

	void UpdateBufferObjects(vtkRenderer *ren, vtkActor *act) override
	{
		if (!m_module->isDirty())
			return;

		if (!m_module->isInitialized())	return;

		auto node = m_module->getParent();

		if (node == NULL || !node->isVisible())	return;
		
		auto mesh = std::dynamic_pointer_cast<dyno::TriangleSet<dyno::DataType3f>>(node->getTopologyModule());
		auto& faces = mesh->getTriangles();
		auto& verts = mesh->getPoints();

		cudaError_t error;

		if (!m_initialized)
		{
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
			this->Primitives[PrimitiveTris].IBO;
			std::vector<unsigned int> indexArray(faces.size() * 3);
			this->Primitives[PrimitiveTris].IBO->Upload(indexArray, vtkOpenGLIndexBufferObject::ElementArrayBuffer);
			this->Primitives[PrimitiveTris].IBO->IndexCount = indexArray.size();
			vtkOpenGLIndexBufferObject* indexBuffer = this->Primitives[PrimitiveTris].IBO;

			// create memory mapper for CUDA
			error = cudaGraphicsGLRegisterBuffer(&m_cudaVBO, vertexBuffer->GetHandle(), cudaGraphicsRegisterFlagsWriteDiscard);
			//printf("%s\n", cudaGetErrorName(error));
			error = cudaGraphicsGLRegisterBuffer(&m_cudaIBO, indexBuffer->GetHandle(), cudaGraphicsRegisterFlagsWriteDiscard);

			// copy index buffer, maybe only need once...
			{
				size_t size;
				void*  cudaPtr = 0;
				error = cudaGraphicsMapResources(1, &m_cudaIBO);
				error = cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, m_cudaIBO);
				error = cudaMemcpy(cudaPtr, faces.begin(), faces.size() * sizeof(unsigned int) * 3, cudaMemcpyDeviceToDevice);
				error = cudaGraphicsUnmapResources(1, &m_cudaIBO);
			}
		}

		// copy vertex memory
		{
			size_t size;
			void*  cudaPtr = 0;

			// upload vertex
			error = cudaGraphicsMapResources(1, &m_cudaVBO);
			//printf("1, %s\n", cudaGetErrorName(error));
			error = cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, m_cudaVBO);
			error = cudaMemcpy(cudaPtr, verts.begin(), verts.size() * sizeof(float) * 3, cudaMemcpyDeviceToDevice);
			error = cudaGraphicsUnmapResources(1, &m_cudaVBO);
		}
	}

private:
	dyno::VtkSurfaceVisualModule* m_module;

	bool				m_initialized = false;

	cudaGraphicsResource*			m_cudaVBO;
	cudaGraphicsResource*			m_cudaIBO;
};

IMPLEMENT_CLASS_COMMON(VtkSurfaceVisualModule, 0)

VtkSurfaceVisualModule::VtkSurfaceVisualModule()
{
	this->setName("surface_renderer");

	m_actor = vtkActor::New();
	m_actor->SetMapper(new SurfaceMapper(this));
}

