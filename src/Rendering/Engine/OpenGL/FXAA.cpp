#include "FXAA.h"
#include <glad/glad.h>
#include "Utility.h"

#include "screen.vert.h"
#include "fxaa.frag.h"

namespace dyno
{
	FXAA::FXAA()
		: RelativeContrastThreshold(1.f / 8.f)
		, HardContrastThreshold(1.f / 16.f)
		, SubpixelBlendLimit(3.f / 4.f)
		, SubpixelContrastThreshold(1.f / 4.f)
		, EndpointSearchIterations(12)
		, UseHighQualityEndpoints(true)
	{
		mScreenQuad = Mesh::ScreenQuad();
		mShaderProgram = Program::createProgramSPIRV(
			SCREEN_VERT, sizeof(SCREEN_VERT),
			FXAA_FRAG, sizeof(FXAA_FRAG));
	}

	FXAA::~FXAA()
	{
		if (mScreenQuad) {
			mScreenQuad->release();
			delete mScreenQuad;
		}
		if (mShaderProgram) {
			mShaderProgram->release();
			delete mShaderProgram;
		}
	}

	void FXAA::apply(int width, int height)
	{
		mShaderProgram->use();

		mShaderProgram->setInt("Input", 1); // GL_TEXTURE1
		mShaderProgram->setVec2("InvTexSize", { 1.f / width, 1.f / height });

		mShaderProgram->setInt("EndpointSearchIterations", EndpointSearchIterations);
		mShaderProgram->setFloat("RelativeContrastThreshold", RelativeContrastThreshold);
		mShaderProgram->setFloat("HardContrastThreshold", HardContrastThreshold);
		mShaderProgram->setFloat("SubpixelBlendLimit", SubpixelBlendLimit);
		mShaderProgram->setFloat("SubpixelContrastThreshold", SubpixelContrastThreshold);
		
		mScreenQuad->draw();
	}
}