#include "GLElementVisualModule.h"

// opengl
#include <glad/glad.h>
#include "GLRenderEngine.h"
#include "Utility.h"

namespace dyno
{
	IMPLEMENT_CLASS(GLElementVisualModule)
	typedef typename TSphere3D<Real> Sphere3D;
	typedef typename TOrientedBox3D<Real> Box3D;
	
	__constant__ int offset3[36][3] = {
		1, 1, 1, //7
		1, -1, 1, //5
		1, 1, -1, //6

		1, -1, 1,//5
		1, -1, -1,//4
		1, 1, -1,//6

		-1, 1, -1,//2
		-1, 1, 1,//3
		1, 1, -1,//6

		1, 1, 1,//7
		1, 1, -1,//6
		-1, 1, 1,//3

		1, 1, 1,//7
		-1, 1, 1,//3
		1, -1, 1,//5

		-1, -1, 1,//1
		1, -1, 1,//5
		-1, 1, 1,//3

		1, -1, 1,//5
		-1, -1, 1,//1
		1, -1, -1,//4

		-1, -1, -1,//0
		1, -1, -1,//4
		-1, -1, 1,//1

		-1, -1, -1,//0
		-1, -1, 1,//1
		-1, 1, -1,//2

		-1, 1, 1,//3
		-1, -1, 1,//1
		-1, 1, -1,//2

		-1, -1, -1,//0
		-1, 1, -1,//2
		1, -1, -1,//4

		1, 1, -1,//6
		1, -1, -1,//4
		-1, 1, -1//2
	};

	GLElementVisualModule::GLElementVisualModule()
	{
		this->setName("element_renderer");
	}

	bool GLElementVisualModule::initializeGL()
	{
		printf("initialize GL\n");
		// create vertex buffer and vertex array object
		mVAO.create();
		mIndexBuffer.create(GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mVertexBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mVAO.bindIndexBuffer(&mIndexBuffer);
		mVAO.bindVertexBuffer(&mVertexBuffer, 0, 3, GL_FLOAT, 0, 0, 0);

		// create shader program
		mShaderProgram = gl::CreateShaderProgram("surface.vert", "surface.frag", "surface.geom");

		printf("aaaaaGL\n");
		return true;
	}

	void GLElementVisualModule::updateStarted()
	{
		std::cout << "Update for GLElementVisualModule started!" << std::endl;
	}

	void GLElementVisualModule::updateEnded()
	{
		std::cout << "Update for GLElementVisualModule ended!" << std::endl;
	}


	__global__ void Mix_Setup_Mapping(
		DArray<int> mapping,
		DArray<int> mapping_shape,
		DArray<int> attribute,
		int num_box,
		int num_sphere,
		int num_tet,
		int num_triangle_sphere
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mapping.size()) return;
		if (pId < num_box * 12)
		{
			mapping[pId] = pId / 12;
			mapping_shape[pId] = pId % 12;
			attribute[pId] = 0;
		}
		else if (pId < num_box * 12 + num_triangle_sphere * num_sphere)
		{
			attribute[pId] = 1;
			int tmp = pId - 12 * num_box;
			mapping[pId] = tmp / num_triangle_sphere;
			mapping_shape[pId] = tmp % num_triangle_sphere;
		}
		else
		{
			attribute[pId] = 2;
			int tmp = pId - (num_box * 12 + num_triangle_sphere * num_sphere);
			mapping[pId] = tmp / 4;
			mapping_shape[pId] = tmp % 4;
		}
	}

	__global__ void Mix_Setup_Box(
		DArray<Box3D> box,
		DArray<Coord3D> centre_box,
		DArray<Coord3D> uu,
		DArray<Coord3D> vv,
		DArray<Coord3D> ww,
		DArray<Coord3D> ex
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= box.size()) return;
		//printf("%d %.3lf %.3lf %.3lf\n", box.size(), box[pId].center[0], box[pId].center[1], box[pId].center[2]);
		centre_box[pId] = Coord3D(box[pId].center[0], box[pId].center[1], box[pId].center[2]);
		uu[pId] = Coord3D(box[pId].u[0], box[pId].u[1], box[pId].u[2]);
		vv[pId] = Coord3D(box[pId].v[0], box[pId].v[1], box[pId].v[2]);
		ww[pId] = Coord3D(box[pId].w[0], box[pId].w[1], box[pId].w[2]);
		ex[pId] = Coord3D(box[pId].extent[0], box[pId].extent[1], box[pId].extent[2]);

		//printf("%.3lf %.3lf %.3lf\n", centre_box[pId].x, centre_box[pId].y, centre_box[pId].z);
	}

	__global__ void Mix_Setup_Sphere(
		DArray<Sphere3D> sphere,
		DArray<float> radius,
		DArray<Coord3D> centre_sphere
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= sphere.size()) return;
		radius[pId] = sphere[pId].radius;
		centre_sphere[pId] = Coord3D(sphere[pId].center[0], sphere[pId].center[1], sphere[pId].center[2]);
	}

	__global__ void Mix_SetupTriangles(
		DArray<Coord3D> vertices,
		DArray<TopologyModule::Triangle> triangles,
		DArray<Coord3D> centre_box,
		DArray<Coord3D> ext,
		DArray<Coord3D> u,
		DArray<Coord3D> v,
		DArray<Coord3D> w,
		DArray<Coord3D> centre_sphere,
		DArray<float> radius,
		DArray<Coord3D> pos,
		DArray<int> index,
		DArray<Tet3D> tets,
		DArray<int> mapping,
		DArray<int> mapping_shape,
		DArray<int> attribute,
		Coord3D color
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mapping.size()) return;
		int idx = mapping_shape[pId];

		Coord3D v1, v2, v3;

		if (attribute[pId] == 0)
		{
			v1 = centre_box[mapping[pId]] + offset3[3 * idx + 0][0] * ext[mapping[pId]][0] * u[mapping[pId]]
				+ offset3[3 * idx + 0][1] * ext[mapping[pId]][1] * v[mapping[pId]]
				+ offset3[3 * idx + 0][2] * ext[mapping[pId]][2] * w[mapping[pId]];

			v2 = centre_box[mapping[pId]] + offset3[3 * idx + 1][0] * ext[mapping[pId]][0] * u[mapping[pId]]
				+ offset3[3 * idx + 1][1] * ext[mapping[pId]][1] * v[mapping[pId]]
				+ offset3[3 * idx + 1][2] * ext[mapping[pId]][2] * w[mapping[pId]];

			v3 = centre_box[mapping[pId]] + offset3[3 * idx + 2][0] * ext[mapping[pId]][0] * u[mapping[pId]]
				+ offset3[3 * idx + 2][1] * ext[mapping[pId]][1] * v[mapping[pId]]
				+ offset3[3 * idx + 2][2] * ext[mapping[pId]][2] * w[mapping[pId]];
		}
		else if (attribute[pId] == 1)
		{
			v1 = centre_sphere[mapping[pId]] + pos[index[3 * idx + 0]] * radius[mapping[pId]];
			v2 = centre_sphere[mapping[pId]] + pos[index[3 * idx + 1]] * radius[mapping[pId]];
			v3 = centre_sphere[mapping[pId]] + pos[index[3 * idx + 2]] * radius[mapping[pId]];
		}
		else
		{
			v1 = Coord3D(tets[mapping[pId]].face(idx).v[0][0], tets[mapping[pId]].face(idx).v[0][1], tets[mapping[pId]].face(idx).v[0][2]);
			v2 = Coord3D(tets[mapping[pId]].face(idx).v[1][0], tets[mapping[pId]].face(idx).v[1][1], tets[mapping[pId]].face(idx).v[1][2]);
			v3 = Coord3D(tets[mapping[pId]].face(idx).v[2][0], tets[mapping[pId]].face(idx).v[2][1], tets[mapping[pId]].face(idx).v[2][2]);
		}


		vertices[3 * pId + 0] = v1;
		vertices[3 * pId + 1] = v2;
		vertices[3 * pId + 2] = v3;
		triangles[pId][0] = 3 * pId + 0;
		triangles[pId][1] = 3 * pId + 1;
		triangles[pId][2] = 3 * pId + 2;
		
	}


	void GLElementVisualModule::updateGL()
	{
		/*auto triSet = this->inTriangleSet()->getDataPtr();

		auto triangles = triSet->getTriangles();
		auto vertices = triSet->getPoints();

		mDrawCount = triangles->size() * 3;

		mVertexBuffer.loadCuda(vertices.begin(), vertices.size() * sizeof(float) * 3);
		mIndexBuffer.loadCuda(triangles->begin(), triangles->size() * sizeof(unsigned int) * 3);*/

		/*Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return;
		}*/

		printf("====================================================== inside box update\n");
		/*m_spheres = discreteSet->getSpheres();
		m_boxes = discreteSet->getBoxes();
		m_tets = discreteSet->getTets();*/

		int num_sphere = 0;// m_spheres.size();
		int num_box = discreteSet->getBoxes().size();
		int num_tet = discreteSet->getTets().size();

		int num_triangle = 100 * num_sphere + 12 * num_box + 4 * num_tet;
		if (vertices.size() != 3 * num_triangle)
		{
			

			vertices.resize(3 * num_triangle);
			

			mapping.resize(num_triangle);
			mapping_shape.resize(num_triangle);
			triangles.resize(num_triangle);
		}

		if (centre_box.size() != num_box)
		{
			centre_box.resize(num_box);
			u.resize(num_box);
			v.resize(num_box);
			w.resize(num_box);
			ext_box.resize(num_box);
		}
		if (centre_sphere.size() != num_sphere)
		{
			radius_sphere.resize(num_sphere);
			centre_sphere.resize(num_sphere);
		}
		uint pDimsBox = cudaGridSize(num_box, BLOCK_SIZE);
		uint pDimsSphere = cudaGridSize(num_sphere, BLOCK_SIZE);

		if (num_triangle > 0)
		{
			cuExecute(
				pDimsBox,
				Mix_Setup_Box,
				discreteSet->getBoxes(),
				centre_box,
				u,
				v,
				w,
				ext_box
				);

			

			/*Mix_Setup_Sphere << <pDimsSphere, BLOCK_SIZE >> > (
				m_spheres,
				radius_sphere,
				centre_sphere
				);
			cuSynchronize();*/
			uint pDims = cudaGridSize(num_triangle, BLOCK_SIZE);
			Mix_Setup_Mapping << <pDims, BLOCK_SIZE >> > (
				mapping,
				mapping_shape,
				attr,
				num_box,
				num_sphere,
				num_tet,
				100
				);
			cuSynchronize();
			Mix_SetupTriangles << <pDims, BLOCK_SIZE >> > (
				vertices,
				triangles,
				centre_box,
				ext_box,
				u,
				v,
				w,
				centre_sphere,
				radius_sphere,
				standard_sphere_position,
				standard_sphere_index,
				discreteSet->getTets(),
				mapping,
				mapping_shape,
				attr,
				Coord3D(0)
				);
			cuSynchronize();


			mDrawCount = num_triangle * 3;

			mVertexBuffer.loadCuda(vertices.begin(), vertices.size() * sizeof(float) * 3);
			mIndexBuffer.loadCuda(triangles.begin(), triangles.size() * sizeof(unsigned int) * 3);

			/*m_triangleRender->setVertexArray(vertices);
			m_triangleRender->setColorArray(colors);
			m_triangleRender->setNormalArray(normals);
			have_triangles = true;*/
		}

	}

	void GLElementVisualModule::paintGL(RenderPass pass)
	{
		mShaderProgram.use();

		unsigned int subroutine;
		if (pass == RenderPass::COLOR)
		{
			mShaderProgram.setVec3("uBaseColor", mBaseColor);
			mShaderProgram.setFloat("uMetallic", mMetallic);
			mShaderProgram.setFloat("uRoughness", mRoughness);
			mShaderProgram.setFloat("uAlpha", mAlpha);	// not implemented!

			subroutine = 0;
			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
		}
		else if (pass == RenderPass::SHADOW)
		{
			subroutine = 1;
			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
		}
		else
		{
			printf("Unknown render pass!\n");
			return;
		}

		mVAO.bind();
		glDrawElements(GL_TRIANGLES, mDrawCount, GL_UNSIGNED_INT, 0);
		mVAO.unbind();

		gl::glCheckError();
	}
}