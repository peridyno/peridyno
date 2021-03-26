#pragma once
#include <GL/glew.h>
#include "cuda_helper_math.h"
#include "ElementRender.h"
#include "Topology/TriangleSet.h"
#include "Vector.h"
#include "Framework/Node.h"
#include "OpenGLContext.h"
#include "Topology/Primitive3D.h"
#include "Topology/DiscreteElements.h"


namespace dyno
{
	typedef typename TSphere3D<Real> Sphere3D;
	typedef typename TOrientedBox3D<Real> Box3D;

	__constant__ int offset3[36][3] = {
		1, 1, 1,
		1, -1, 1,
		1, 1, -1,

		1, -1, 1,
		1, -1, -1,
		1, 1, -1,

		-1, 1, -1,
		-1, 1, 1,
		1, 1, -1,

		1, 1, 1,
		1, 1, -1,
		-1, 1, 1,

		1, 1, 1,
		-1, 1, 1,
		1, -1, 1,

		-1, -1, 1,
		1, -1, 1,
		-1, 1, 1,

		1, -1, 1,
		-1, -1, 1,
		1, -1, -1,

		-1, -1, -1,
		1, -1, -1,
		-1, -1, 1,

		-1, -1, -1,
		-1, -1, 1,
		-1, 1, -1,

		-1, 1, 1,
		-1, -1, 1,
		-1, 1, -1,

		-1, -1, -1,
		-1, 1, -1,
		1, -1, -1,

		1, 1, -1,
		1, -1, -1,
		-1, 1, -1
	};

	IMPLEMENT_CLASS(ElementRender)

		ElementRender::ElementRender()
		: VisualModule()
		, m_color(Vector3f(0.2f, 0.3, 0.0f))
	{
	}

	ElementRender::~ElementRender()
	{
		vertices.clear();
		normals.clear();
		colors.clear();
	}

	bool ElementRender::initializeImpl()
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return false;
		}
		num_triangle_sphere = 0;

		std::vector<float3> sphereVertices;
		std::vector<int> sphereIndices;

		for (int y = 0; y <= y_segments; y++)
		{
			for (int x = 0; x <= x_segments; x++)
			{
				float xSegment = (float)x / (float)x_segments;
				float ySegment = (float)y / (float)y_segments;
				float xPos = std::cos(xSegment * 2.0f * M_PI) * std::sin(ySegment * M_PI);
				float yPos = std::cos(ySegment * M_PI);
				float zPos = std::sin(xSegment * 2.0f * M_PI) * std::sin(ySegment * M_PI);
				sphereVertices.push_back(make_float3(xPos, yPos, zPos));
			}
		}


		for (int i = 0; i<y_segments; i++)
		{
			for (int j = 0; j<x_segments; j++)
			{
				sphereIndices.push_back(i * (x_segments + 1) + j);
				sphereIndices.push_back((i + 1) * (x_segments + 1) + j);
				sphereIndices.push_back((i + 1) * (x_segments + 1) + j + 1);
				sphereIndices.push_back(i* (x_segments + 1) + j);
				sphereIndices.push_back((i + 1) * (x_segments + 1) + j + 1);
				sphereIndices.push_back(i * (x_segments + 1) + j + 1);
				num_triangle_sphere += 2;
			}
		}

		//printf("NUM_TRIANGLE: %d %d\n", num_triangle, sphereIndices.size());

		standard_sphere_position.resize(sphereVertices.size());
		standard_sphere_index.resize(sphereIndices.size());

		standard_sphere_position.assign(sphereVertices);
		standard_sphere_index.assign(sphereIndices);




		m_triangleRender = std::make_shared<TriangleRender>();


		sphereVertices.clear();
		sphereIndices.clear();

	}


	


	__global__ void Mix_Setup_Mapping(
		DArray<int> mapping,
		DArray<int> mapping_shape,
		DArray<int> attribute,
		int num_box,
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
		else
		{
			attribute[pId] = 1;
			int tmp = pId - 12 * num_box;
			mapping[pId] = tmp / num_triangle_sphere;
			mapping_shape[pId] = tmp% num_triangle_sphere;
		}
	}

	__global__ void Mix_Setup_Box(
		DArray<Box3D> box,
		DArray<float3> centre_box,
		DArray<float3> uu,
		DArray<float3> vv,
		DArray<float3> ww,
		DArray<float3> ex
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= box.size()) return;
		//printf("%d %.3lf %.3lf %.3lf\n", box.size(), box[pId].center[0], box[pId].center[1], box[pId].center[2]);
		centre_box[pId] = make_float3(box[pId].center[0], box[pId].center[1], box[pId].center[2]);
		uu[pId] = make_float3(box[pId].u[0], box[pId].u[1], box[pId].u[2]);
		vv[pId] = make_float3(box[pId].v[0], box[pId].v[1], box[pId].v[2]);
		ww[pId] = make_float3(box[pId].w[0], box[pId].w[1], box[pId].w[2]);
		ex[pId] = make_float3(box[pId].extent [0], box[pId].extent[1], box[pId].extent[2]);

		//printf("%.3lf %.3lf %.3lf\n", centre_box[pId].x, centre_box[pId].y, centre_box[pId].z);
	}

	__global__ void Mix_Setup_Sphere(
		DArray<Sphere3D> sphere,
		DArray<float> radius,
		DArray<float3> centre_sphere
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= sphere.size()) return;
		radius[pId] = sphere[pId].radius;
		centre_sphere[pId] = make_float3(sphere[pId].center[0], sphere[pId].center[1], sphere[pId].center[2]);	
	}

	__global__ void Mix_SetupTriangles(
		DArray<float3> vertices,
		DArray<float3> normals,
		DArray<float3> colors,
		DArray<float3> centre_box,
		DArray<float3> ext,
		DArray<float3> u,
		DArray<float3> v,
		DArray<float3> w,
		DArray<float3> centre_sphere,
		DArray<float> radius,
		DArray<float3> pos,
		DArray<int> index,
		DArray<int> mapping,
		DArray<int> mapping_shape,
		DArray<int> attribute,
		float3 color
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= mapping.size()) return;
		int idx = mapping_shape[pId];

		float3 v1, v2, v3;

		if (attribute[pId] == 0)
		{
			v1 = centre_box[mapping[pId]] + offset3[3 * idx + 0][0] * ext[mapping[pId]].x * u[mapping[pId]]
				+ offset3[3 * idx + 0][1] * ext[mapping[pId]].y * v[mapping[pId]]
				+ offset3[3 * idx + 0][2] * ext[mapping[pId]].z * w[mapping[pId]];

			v2 = centre_box[mapping[pId]] + offset3[3 * idx + 1][0] * ext[mapping[pId]].x * u[mapping[pId]]
				+ offset3[3 * idx + 1][1] * ext[mapping[pId]].y * v[mapping[pId]] 
				+ offset3[3 * idx + 1][2] * ext[mapping[pId]].z * w[mapping[pId]];

			v3 = centre_box[mapping[pId]] + offset3[3 * idx + 2][0] * ext[mapping[pId]].x * u[mapping[pId]] 
				+ offset3[3 * idx + 2][1] * ext[mapping[pId]].y * v[mapping[pId]] 
				+ offset3[3 * idx + 2][2] * ext[mapping[pId]].z * w[mapping[pId]];
		}
		else
		{
			v1 = centre_sphere[mapping[pId]] + pos[index[3 * idx + 0]] * radius[mapping[pId]];
			v2 = centre_sphere[mapping[pId]] + pos[index[3 * idx + 1]] * radius[mapping[pId]];
			v3 = centre_sphere[mapping[pId]] + pos[index[3 * idx + 2]] * radius[mapping[pId]];
		}



		vertices[3 * pId + 0] = v1;
		vertices[3 * pId + 1] = v2;
		vertices[3 * pId + 2] = v3;

		float3 triN = cross(v2 - v1, v3 - v1);
		triN = normalize(triN);

		normals[3 * pId + 0] = triN;
		normals[3 * pId + 1] = triN;
		normals[3 * pId + 2] = triN;

		colors[3 * pId + 0] = color;
		colors[3 * pId + 1] = color;
		colors[3 * pId + 2] = color;
	}

	void ElementRender::updateRenderingContext()
	{
		
		Node* parent = getParent();
		if (parent == NULL)
		{
		Log::sendMessage(Log::Error, "Should insert this module into a node!");
		return;
		}

		auto discreteSet = TypeInfo::cast<DiscreteElements<DataType3f>>(parent->getTopologyModule());
		if (discreteSet == nullptr)
		{
		Log::sendMessage(Log::Error, "DiscreteElements: The topology module is not supported!");
		return;
		}
		
		//printf("====================================================== inside box update\n");
		m_spheres = discreteSet->getSpheres();
		m_boxes = discreteSet->getBoxes();

		int num_sphere = m_spheres.size();
		int num_box = m_boxes.size();

		//printf("%d %d\n", num_sphere,num_box);



		int num_triangle = num_triangle_sphere * num_sphere + 12 * num_box;
		if (vertices.size() != 3 * num_triangle)
		{
			m_triangleRender->resize(num_triangle);

			vertices.resize(3 * num_triangle);
			normals.resize(3 * num_triangle);
			colors.resize(3 * num_triangle);

			mapping.resize(num_triangle);
			mapping_shape.resize(num_triangle);
			attr.resize(num_triangle);
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


		Mix_Setup_Box << <pDimsBox, BLOCK_SIZE >> > (
			m_boxes,
			centre_box,
			u,
			v,
			w,
			ext_box
			);

		cuSynchronize();
		
		Mix_Setup_Sphere << <pDimsSphere, BLOCK_SIZE >> > (
			m_spheres,
			radius_sphere,
			centre_sphere
			);
		cuSynchronize();
		uint pDims = cudaGridSize(num_triangle, BLOCK_SIZE);
		Mix_Setup_Mapping << <pDims, BLOCK_SIZE >> > (
			mapping,
			mapping_shape,
			attr,
			num_box,
			num_triangle_sphere
			);
		cuSynchronize();
		Mix_SetupTriangles << <pDims, BLOCK_SIZE >> > (
			vertices,
			normals,
			colors,
			centre_box,
			ext_box,
			u,
			v,
			w,
			centre_sphere,
			radius_sphere,
			standard_sphere_position,
			standard_sphere_index,
			mapping,
			mapping_shape,
			attr,
			make_float3(m_color[0], m_color[1], m_color[2])
			);
		cuSynchronize();
		

		m_triangleRender->setVertexArray(vertices);
		m_triangleRender->setColorArray(colors);
		m_triangleRender->setNormalArray(normals);
	}

	void ElementRender::display()
	{
		//printf("======================================   inside box rendering\n");
		glMatrixMode(GL_MODELVIEW_MATRIX);
		glPushMatrix();

		glRotatef(m_rotation.x(), m_rotation.y(), m_rotation.z(), m_rotation.w());
		glTranslatef(m_translation[0], m_translation[1], m_translation[2]);
		glScalef(m_scale[0], m_scale[1], m_scale[2]);

		m_triangleRender->display();

		glPopMatrix();
	}

	void ElementRender::setColor(Vector3f color)
	{
		m_color = color;
	}

}