#include "GLWireframeVisualModule.h"

// opengl
#include <glad/glad.h>
#include "GLRenderEngine.h"
#include "Utility.h"

namespace dyno
{
	IMPLEMENT_CLASS(GLWireframeVisualModule)

	GLWireframeVisualModule::GLWireframeVisualModule()
	{
		this->setName("wireframe_renderer");
		this->varRadius()->setRange(0.001, 0.01);
	}


	std::string GLWireframeVisualModule::caption()
	{
		return "Wireframe Visual Module";
	}

	bool GLWireframeVisualModule::initializeGL()
	{
		createCylinder();

		mEdges.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		mPoints.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);

		// create shader program
		mShaderProgram = gl::ShaderFactory::createShaderProgram("line.vert", "surface.frag");
		
		return true;
	}


	void GLWireframeVisualModule::updateGL()
	{
		auto edgeSet = this->inEdgeSet()->getDataPtr();

		auto& edges = edgeSet->getEdges();
		auto& vertices = edgeSet->getPoints();

		mNumEdges = edges.size();

		mPoints.loadCuda(vertices.begin(), vertices.size() * sizeof(float) * 3);
		mEdges.loadCuda(edges.begin(), edges.size() * sizeof(unsigned int) * 2);
	}

	void GLWireframeVisualModule::paintGL(GLRenderPass pass)
	{
		if (mNumEdges == 0)
			return;

		mShaderProgram.use();

		unsigned int subroutine = (unsigned int)pass;

		glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);

		if (pass == GLRenderPass::COLOR)
		{
			mShaderProgram.setVec3("uBaseColor", this->varBaseColor()->getData());
			mShaderProgram.setFloat("uMetallic", this->varMetallic()->getData());
			mShaderProgram.setFloat("uRoughness", this->varRoughness()->getData());
			mShaderProgram.setFloat("uAlpha", this->varAlpha()->getData());
		}
		else if (pass == GLRenderPass::SHADOW)
		{
			// lines should cast shadow?
		}
		else
		{
			printf("Unknown render pass!\n");
			return;
		}

		mShaderProgram.setFloat("uRadius", this->varRadius()->getData());

		mPoints.bindBufferBase(1);
		mEdges.bindBufferBase(2);

		mCylinder.vao.bind();
		glDrawElementsInstanced(GL_TRIANGLES, mCylinder.drawCount, GL_UNSIGNED_INT, 0, mNumEdges);
		mCylinder.vao.unbind();

		gl::glCheckError();
	}



	void GLWireframeVisualModule::createCylinder()
	{
		// create vertex buffer and vertex array object
		mCylinder.vao.create();

		mCylinder.vertices.create(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
		mCylinder.normals.create(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
		mCylinder.indices.create(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);

		mCylinder.vao.bindIndexBuffer(&mCylinder.indices);
		mCylinder.vao.bindVertexBuffer(&mCylinder.vertices, 0, 3, GL_FLOAT, 0, 0, 0);
		mCylinder.vao.bindVertexBuffer(&mCylinder.normals, 1, 3, GL_FLOAT, 0, 0, 0);

		float sectorStep = 2 * M_PI / mCylinder.nSectors;

		std::vector<glm::vec3> vertices;
		std::vector<glm::vec3> normals;
		std::vector<glm::ivec3> indices;

		for (int i = 0; i <= mCylinder.nSectors; ++i) {
			float sectorAngle = i * sectorStep;
			float x = cosf(sectorAngle);
			float y = sinf(sectorAngle);
			
			vertices.push_back({ x, y, 0.f });
			vertices.push_back({ x, y, 1.f });
			normals.push_back({ x, y, 0.f });
			normals.push_back({ x, y, 0.f });
		}

		for (int i = 0; i < mCylinder.nSectors; ++i) {
			int offset = i * 2;
			indices.push_back({ offset,     offset + 1, offset + 2 });
			indices.push_back({ offset + 2, offset + 1, offset + 3 });
		}

		// base and top disk
		{
			int offset0 = vertices.size();

			for (int i = 0; i <= mCylinder.nSectors; ++i) {
				float sectorAngle = i * sectorStep;
				float x = cosf(sectorAngle);
				float y = sinf(sectorAngle);

				vertices.push_back({ x, y, 0.f });
				vertices.push_back({ x, y, 1.f });
				normals.push_back({ 0, 0, -1 });
				normals.push_back({ 0, 0,  1 });
			}

			int index0 = vertices.size();
			vertices.push_back({ 0, 0, 0 });
			normals.push_back({ 0, 0, -1 });

			int index1 = vertices.size();
			vertices.push_back({ 0, 0, 1 });
			normals.push_back({ 0, 0, 1 });

			for (int i = 0; i < mCylinder.nSectors; ++i) {
				int offset = offset0 + i * 2;
				indices.push_back({ index0, offset + 0, offset + 2 });
				indices.push_back({ index1, offset + 1, offset + 3 });
			}
		}


		mCylinder.vertices.load(vertices.data(), sizeof(glm::vec3) * vertices.size());
		mCylinder.normals.load(normals.data(), sizeof(glm::vec3) * normals.size());
		mCylinder.indices.load(indices.data(), sizeof(glm::ivec3) * indices.size());

		mCylinder.drawCount = indices.size() * 3;
	}
}