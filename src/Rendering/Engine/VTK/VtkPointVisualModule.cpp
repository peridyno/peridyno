#include "VtkPointVisualModule.h"

// framework
#include <Node.h>
#include <SceneGraph.h>
#include <Topology/TriangleSet.h>

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

#include "SceneGraphFactory.h"

#ifdef _WIN32
#include <windows.h>
#endif
#include <cuda_gl_interop.h>

using namespace dyno;

class PointMapper : public vtkOpenGLPolyDataMapper
{
public:
	PointMapper(VtkPointVisualModule* v): m_module(v)
	{
		// create psedo data, required by the vtkOpenGLPolyDataMapper to render content
		vtkNew<vtkPoints> points;

		auto scn = dyno::SceneGraphFactory::instance()->active();
		Vec3f bbox0 = scn->getLowerBound();
		Vec3f bbox1 = scn->getUpperBound();
		points->InsertNextPoint(bbox0[0], bbox0[1], bbox0[2]);
		points->InsertNextPoint(bbox1[0], bbox1[1], bbox1[2]);

		vtkNew<vtkPolyData> polyData;
		polyData->SetPoints(points);
		SetInputData(polyData);
	}
	

	void UpdateBufferObjects(vtkRenderer *ren, vtkActor *act) override
	{
		if (!m_module->isDirty())
			return;

		if (!m_module->isInitialized())	return;

		auto node = m_module->getParent();

		if (node == NULL || !node->isVisible())	return;
		
		auto pSet = m_module->inPointSet()->getDataPtr();// std::dynamic_pointer_cast<dyno::PointSet<dyno::DataType3f>>(node->getTopologyModule());
		auto verts = pSet->getPoints();

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
	dyno::VtkPointVisualModule* m_module;

	bool				m_initialized = false;

	cudaGraphicsResource*			m_cudaVBO;
};

IMPLEMENT_CLASS_COMMON(VtkPointVisualModule, 0)

VtkPointVisualModule::VtkPointVisualModule()
{
	this->setName("point_renderer");

	m_actor = vtkActor::New();
	m_actor->GetProperty()->SetRepresentationToPoints();
	m_actor->GetProperty()->RenderPointsAsSpheresOn();
	m_actor->GetProperty()->SetPointSize(2.0);
	m_actor->SetMapper(new PointMapper(this));
}

