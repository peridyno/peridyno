

#pragma once
#include "Topology/TriangleSet.h"
#include "Topology/DiscreteElements.h"

#include "GLVisualModule.h"
#include "GLCudaBuffer.h"
#include "gl/VertexArray.h"
#include "gl/Program.h"


namespace dyno
{
	template<typename TDataType> class DiscreteElements;
	class GLElementVisualModule : public GLVisualModule
	{
		DECLARE_CLASS(GLElementVisualModule)
	public:
		GLElementVisualModule();

	public:
		std::shared_ptr<DiscreteElements<DataType3f>> discreteSet;
		//DEF_INSTANCE_IN(TriangleSet<DataType3f>, TriangleSet, "");

		DEF_VAR_IN(float, TimeStep, "dt");

	protected:
		virtual void paintGL(RenderPass mode) override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;

		void updateStarted() override;
		void updateEnded() override;

	private:

		gl::Program mShaderProgram;
		gl::VertexArray	mVAO;

		GLCudaBuffer	mVertexBuffer;
		GLCudaBuffer 	mIndexBuffer;

		unsigned int	mDrawCount = 0;

		DArray<TopologyModule::Triangle> triangles;
		DArray<Coord3D> vertices;
		DArray<int> mapping;
		DArray<int> mapping_shape;

		DArray<Coord3D> centre_sphere;
		DArray<float> radius_sphere;

		DArray<Coord3D> centre_box;
		DArray<Coord3D> u;
		DArray<Coord3D> v;
		DArray<Coord3D> w;
		DArray<Coord3D> ext_box;

		DArray<Coord3D> standard_sphere_position;
		DArray<int> standard_sphere_index;
		DArray<int> attr;
	};
};