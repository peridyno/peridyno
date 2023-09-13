#include "Material.h"

namespace gl
{
	Material::Material()
	{
	}

	Material::~Material()
	{
	}

	void Material::create()
	{
	}

	void Material::release()
	{

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