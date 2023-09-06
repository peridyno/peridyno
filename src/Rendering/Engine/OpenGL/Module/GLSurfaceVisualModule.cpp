#include "GLSurfaceVisualModule.h"
#include "Utility.h"

#include <glad/glad.h>

#include "surface.vert.h"
#include "surface.frag.h"
#include "surface.geom.h"

namespace dyno
{
	IMPLEMENT_CLASS(GLSurfaceVisualModule)

	GLSurfaceVisualModule::GLSurfaceVisualModule()
	{
		this->setName("surface_renderer");

		this->inColor()->tagOptional(true);

		this->inNormal()->tagOptional(true);
		this->inNormalIndex()->tagOptional(true);
		this->inTexCoord()->tagOptional(true);
		this->inTexCoordIndex()->tagOptional(true);

#ifdef CUDA_BACKEND
		this->inColorTexture()->tagOptional(true); 
#endif
	}

	GLSurfaceVisualModule::~GLSurfaceVisualModule()
	{
// 		mIndexBuffer.release();
// 		mVertexBuffer.release();
// 		mNormalBuffer.release();
// 		mColorBuffer.release();
// 
// 		triangles.clear();
// 		vertices.clear();
// 		normals.clear();
// 		colors.clear();
	}

	std::string GLSurfaceVisualModule::caption()
	{
		return "Surface Visual Module";
	}

	bool GLSurfaceVisualModule::initializeGL()
	{
		// create vertex buffer and vertex array object
		mVAO.create();

		mVertexIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mNormalIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mTexCoordIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mVertexPosition.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		mVertexColor.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		mNormal.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		mTexCoord.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);

		// create shader program
		mShaderProgram = gl::Program::createProgramSPIRV(
			SURFACE_VERT, sizeof(SURFACE_VERT),
			SURFACE_FRAG, sizeof(SURFACE_FRAG),
			SURFACE_GEOM, sizeof(SURFACE_GEOM));

		// create shader uniform buffer
		mRenderParamsUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		mPBRMaterialUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		return true;
	}

	void GLSurfaceVisualModule::releaseGL()
	{
		mShaderProgram->release();
		delete mShaderProgram;
		mShaderProgram = 0;

		// vertex array object
		mVAO.release();

		// vertex array buffer
		mVertexIndex.release();
		mNormalIndex.release();
		mTexCoordIndex.release();

		// shader storage buffer
		mVertexColor.release();
		mVertexPosition.release();
		mNormal.release();
		mTexCoord.release();

		// release uniform block
		mRenderParamsUBlock.release();
		mPBRMaterialUBlock.release();
	}

	void GLSurfaceVisualModule::updateGL()
	{
		mNumTriangles = mVertexIndex.count();
		if (mNumTriangles == 0) return;

		mVertexIndex.updateGL();

		// setup VAO binding...
		mVAO.bind();

		// vertex index
		{
			mVertexIndex.bind();
			glEnableVertexAttribArray(0);
			glVertexAttribIPointer(0, 1, GL_INT, sizeof(int), (void*)0);
		}

		// normal
		{
			if (mNormalIndex.count() == mNumTriangles) {
				mNormalIndex.updateGL();
				mNormalIndex.bind();
				glEnableVertexAttribArray(1);
				glVertexAttribIPointer(1, 1, GL_INT, sizeof(int), (void*)0);
			}
			else
			{
				glDisableVertexAttribArray(1);
				glVertexAttribI4i(1, -1, -1, -1, -1);
			}
		}

		// texcoord
		{
			if (mTexCoordIndex.count() == mNumTriangles) {
				mTexCoordIndex.updateGL();
				mTexCoordIndex.bind();
				glEnableVertexAttribArray(2);
				glVertexAttribIPointer(2, 1, GL_INT, sizeof(int), (void*)0);
			}
			else
			{
				glDisableVertexAttribArray(2);
				glVertexAttribI4i(2, -1, -1, -1, -1);
			}
		}

		// instance transforms
		glDisableVertexAttribArray(3);
		glDisableVertexAttribArray(4);
		glDisableVertexAttribArray(5);
		glDisableVertexAttribArray(6);
		glDisableVertexAttribArray(7);
		glDisableVertexAttribArray(8);

		mVAO.unbind();

		// update shader storage buffer
		mVertexPosition.updateGL();

		// vertex color
		if (this->varColorMode()->currentKey() == EColorMode::CM_Vertex) {
			mVertexColor.updateGL();
		}
		// vertex normal
		if(this->varUseVertexNormal()->getValue()) {
			mNormal.updateGL();
		}

		// texture coordinates
		if (mTexCoord.count() > 0) {
			mTexCoord.updateGL();
		}

#ifdef CUDA_BACKEND
		// update texture content
		mColorTexture.updateGL();
#endif

		gl::glCheckError();
	}

	void GLSurfaceVisualModule::updateImpl()
	{
		// update data
		auto triSet = this->inTriangleSet()->constDataPtr();
		auto indices = triSet->getTriangles();
		auto vertices = triSet->getPoints();

		mVertexIndex.load(indices);
		mVertexPosition.load(vertices);

		if (this->varColorMode()->getValue() == EColorMode::CM_Vertex &&
			!this->inColor()->isEmpty() &&
			this->inColor()->getDataPtr()->size() == vertices.size())
		{
			auto colors = this->inColor()->getData();
			mVertexColor.load(colors);
		}

		// generate per-vertex normal
		if (this->varUseVertexNormal()->getValue())
		{
#ifdef CUDA_BACKEND
			//TODO: optimize the performance
			if (this->inNormal()->isEmpty()) {
				triSet->update();
				auto normals = triSet->getVertexNormals();
				mNormal.load(normals);
			}
			else
			{
				mNormal.load(this->inNormal()->constData());
				// has separate normal index?
				if (!this->inNormalIndex()->isEmpty())
					mNormalIndex.load(this->inNormalIndex()->constData());
			}
#endif
		}

		// texture coordinates
		{
			if (!this->inTexCoord()->isEmpty()) {
				mTexCoord.load(this->inTexCoord()->constData());
			}

			if (!this->inTexCoordIndex()->isEmpty()) {
				mTexCoordIndex.load(this->inTexCoordIndex()->constData());
			}
		}

#ifdef CUDA_BACKEND
		// texture
		if (!inColorTexture()->isEmpty()) {
			mColorTexture.load(inColorTexture()->constData());
		}
#endif

	}

	void GLSurfaceVisualModule::paintGL(const RenderParams& rparams)
	{
		if (mNumTriangles == 0)
			return;

		mShaderProgram->use();

		if (rparams.mode == GLRenderMode::COLOR) {
		}
		else if (rparams.mode == GLRenderMode::SHADOW) {
		}
		else if (rparams.mode == GLRenderMode::TRANSPARENCY) {
		}
		else {
			printf("GLSurfaceVisualModule: Unknown render mode!\n");
			return;
		}

		// bind vertex data
		mVertexPosition.bindBufferBase(8);
		mNormal.bindBufferBase(9);
		mTexCoord.bindBufferBase(10);
		mVertexColor.bindBufferBase(11);

		// setup uniforms
		struct {
			glm::vec3 color;
			float metallic;
			float roughness;
			float alpha;
		} pbr;

		auto color = this->varBaseColor()->getValue();
		pbr.color = { color.r, color.g, color.b };
		pbr.metallic = this->varMetallic()->getValue();
		pbr.roughness = this->varRoughness()->getValue();
		pbr.alpha = this->varAlpha()->getValue();

		mShaderProgram->setInt("uVertexNormal", this->varUseVertexNormal()->getValue());
		mShaderProgram->setInt("uColorMode", this->varColorMode()->currentKey());
		mShaderProgram->setInt("uInstanced", mInstanceCount > 0);

		// setup uniform buffer
		mRenderParamsUBlock.load((void*)&rparams, sizeof(RenderParams));
		mRenderParamsUBlock.bindBufferBase(0);

		mPBRMaterialUBlock.load((void*)&pbr, sizeof(pbr));
		mPBRMaterialUBlock.bindBufferBase(1);


#ifdef CUDA_BACKEND
		// bind texture
		if (mColorTexture.isValid()) {
			mColorTexture.bind(GL_TEXTURE10);
		}
		else
#endif
		{
			glActiveTexture(GL_TEXTURE10);
			glBindTexture(GL_TEXTURE_2D, 0);
		}
		mVAO.bind();

		if(mInstanceCount > 0)
			glDrawArraysInstanced(GL_TRIANGLES, 0, mNumTriangles * 3, mInstanceCount);
		else
			glDrawArrays(GL_TRIANGLES, 0, mNumTriangles * 3);

		gl::glCheckError();
		mVAO.unbind();
	}
}