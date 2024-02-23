#include "Material.h"

namespace dyno::render
{
	Material::Material() : texColor()
	{
	}

	Material::~Material()
	{
		texColor.clear();
		texBump.clear();
	}

	void Material::create()
	{
	}

	void Material::release()
	{
		mColorTexture.release();
		mBumpTexture.release();
	}

	void Material::update()
	{
		if (texColor.size())
		{
			mColorTexture.load(texColor);
		}

		if (texBump.size())
		{
			mBumpTexture.load(texBump);
		}
	}

	void Material::updateGL()
	{
		mColorTexture.updateGL();
		mBumpTexture.updateGL();
	}

}