#pragma once

#include <Framework/ModuleVisual.h>
#include <glm/vec3.hpp>

namespace dyno
{
	class RenderEngine;
	class GLVisualModule : public VisualModule
	{
	public:
		enum ShadowMode
		{
			NONE = 0,	// do not cast/receive shadow
			CAST = 1,	// cast shadow
			RECV = 2,	// receive shadow
			ALL = 3,	// both...
		};

		enum ColorMapMode
		{
			CONSTANT = 0,	// use constant color
			VELOCITY_JET = 1,
			VELOCITY_HEAT = 2,
			FORCE_JET = 3,
			FORCE_HEAT = 4,
		};

	public:
		GLVisualModule();

		// override
		void display() final;
		void updateRenderingContext() final;

		// material properties
		void setColor(const glm::vec3& color);
		void setMetallic(float metallic);
		void setRoughness(float roughness);
		void setAlpha(float alpha);

		// colormap
		void setColorMapMode(ColorMapMode mode = CONSTANT);
		void setColorMapRange(float vmin, float vmax);

		// shadow mode
		void setShadowMode(ShadowMode mode);

		virtual bool isTransparent() const;

	protected:
		virtual bool initializeGL() = 0;
		virtual void updateGL() = 0;
		virtual void paintGL() = 0;

	private:
		bool isGLInitialized = false;

	protected:

		// material properties
		glm::vec3		mBaseColor = glm::vec3(0.8f);
		float			mMetallic = 0.5f;
		float			mRoughness = 0.5f;
		float			mAlpha = 1.f;

		// color map
		ColorMapMode	mColorMode = CONSTANT;
		float			mColorMin = 0.f;
		float			mColorMax = 1.f;

		ShadowMode		mShadowMode;

		friend class RenderEngine;
	};
};