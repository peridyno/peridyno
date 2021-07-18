#include "Mesh.h"

#include <glad/glad.h>
#include <vector>

namespace gl
{

	void Mesh::create()
	{
		VertexArray::create();
		mIndexBuffer.create(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
		mVertexBuffer.create(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
		bindIndexBuffer(&mIndexBuffer);
		bindVertexBuffer(&mVertexBuffer, 0, 3, GL_FLOAT, 0, 0, 0);
	}

	void Mesh::release()
	{
		unbind();
		// release buffers
		mVertexBuffer.release();
		mIndexBuffer.release();

		// release VAO
		VertexArray::release();
	}


	void Mesh::draw(int instance)
	{
		this->bind();

		if (instance > 0)
			glDrawElementsInstanced(GL_TRIANGLES, mDrawCount, GL_UNSIGNED_INT, 0, instance);
		else
			glDrawElements(GL_TRIANGLES, mDrawCount, GL_UNSIGNED_INT, 0);
	}

	// helper functions
	Mesh Mesh::Sphere(float radius, int sectorCount, int stackCount)
	{	// create sphere
		// http://www.songho.ca/opengl/gl_sphere.html

		std::vector<float> vertices;
		std::vector<float> normals;
		std::vector<float> texCoords;
		std::vector<int>   indices;

		float x, y, z, xy;                              // vertex position
		float nx, ny, nz, lengthInv = 1.0f / radius;    // vertex normal
		float s, t;                                     // vertex texCoord

		const float PI = 3.14159265f;
		float sectorStep = 2 * PI / sectorCount;
		float stackStep = PI / stackCount;
		float sectorAngle, stackAngle;

		for (int i = 0; i <= stackCount; ++i)
		{
			stackAngle = PI / 2 - i * stackStep;        // starting from pi/2 to -pi/2
			xy = radius * cosf(stackAngle);             // r * cos(u)
			z = radius * sinf(stackAngle);              // r * sin(u)

			// add (sectorCount+1) vertices per stack
			// the first and last vertices have same position and normal, but different tex coords
			for (int j = 0; j <= sectorCount; ++j)
			{
				sectorAngle = j * sectorStep;           // starting from 0 to 2pi

				// vertex position (x, y, z)
				x = xy * cosf(sectorAngle);             // r * cos(u) * cos(v)
				y = xy * sinf(sectorAngle);             // r * cos(u) * sin(v)
				vertices.push_back(x);
				vertices.push_back(y);
				vertices.push_back(z);

				// normalized vertex normal (nx, ny, nz)
				nx = x * lengthInv;
				ny = y * lengthInv;
				nz = z * lengthInv;
				normals.push_back(nx);
				normals.push_back(ny);
				normals.push_back(nz);

				// vertex tex coord (s, t) range between [0, 1]
				s = (float)j / sectorCount;
				t = (float)i / stackCount;
				texCoords.push_back(s);
				texCoords.push_back(t);
			}
		}

		// generate CCW index list of sphere triangles
		int k1, k2;
		for (int i = 0; i < stackCount; ++i)
		{
			k1 = i * (sectorCount + 1);     // beginning of current stack
			k2 = k1 + sectorCount + 1;      // beginning of next stack

			for (int j = 0; j < sectorCount; ++j, ++k1, ++k2)
			{
				// 2 triangles per sector excluding first and last stacks
				// k1 => k2 => k1+1
				if (i != 0)
				{
					indices.push_back(k1);
					indices.push_back(k2);
					indices.push_back(k1 + 1);
				}

				// k1+1 => k2 => k2+1
				if (i != (stackCount - 1))
				{
					indices.push_back(k1 + 1);
					indices.push_back(k2);
					indices.push_back(k2 + 1);
				}
			}
		}

		Mesh vao;
		vao.create();
		vao.mVertexBuffer.load(vertices.data(), vertices.size() * sizeof(float));
		vao.mIndexBuffer.load(indices.data(), indices.size() * sizeof(unsigned int));
		vao.mDrawCount = indices.size();
		return vao;
	}

	Mesh Mesh::AABB(glm::vec3 p0, glm::vec3 p1)
	{
		std::vector<float> vertices = {
			p0.x, p0.y, p0.z,
			p0.x, p0.y, p1.z,
			p1.x, p0.y, p1.z,
			p1.x, p0.y, p0.z,

			p0.x, p1.y, p0.z,
			p0.x, p1.y, p1.z,
			p1.x, p1.y, p1.z,
			p1.x, p1.y, p0.z,
		};

		std::vector<unsigned int> indices = {
			0, 1, 2, 2, 3, 0, // bottom
			7, 6, 5, 5, 4, 7, // top
			4, 5, 1, 1, 0, 4, // left
			3, 2, 6, 6, 7, 3, // right
			0, 3, 7, 7, 4, 0, // front
			5, 6, 2, 2, 1, 5, // back
		};

		Mesh vao;
		vao.create();
		vao.mVertexBuffer.load(vertices.data(), vertices.size() * sizeof(float));
		vao.mIndexBuffer.load(indices.data(), indices.size() * sizeof(unsigned int));
		vao.mDrawCount = indices.size();
		return vao;
	}

	Mesh Mesh::ScreenQuad()
	{
		std::vector<float> vertices = {
			-1, -1, 0,
			-1,  1, 0,
			 1, -1, 0,
			 1,  1, 0
		};

		std::vector<unsigned int> indices = {
			0, 2, 1,
			2, 3, 1
		};

		Mesh vao;
		vao.create();
		vao.mVertexBuffer.load(vertices.data(), vertices.size() * sizeof(float));
		vao.mIndexBuffer.load(indices.data(), indices.size() * sizeof(unsigned int));
		vao.mDrawCount = indices.size();
		return vao;
	}

	Mesh Mesh::Plane(float scale)
	{
		std::vector<float> vertices = {
			-scale, 0, -scale,
			-scale, 0,  scale,
			 scale, 0, -scale,
			 scale, 0,  scale,
		};

		std::vector<unsigned int> indices = {
			0, 1, 2,
			2, 1, 3
		};

		Mesh vao;
		vao.create();
		vao.mVertexBuffer.load(vertices.data(), vertices.size() * sizeof(float));
		vao.mIndexBuffer.load(indices.data(), indices.size() * sizeof(unsigned int));
		vao.mDrawCount = indices.size();
		return vao;
	}
}