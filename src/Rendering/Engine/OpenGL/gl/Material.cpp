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
		if (!texColor.isEmpty())
		{
			mColorTexture.load(texColor);
		}

		if (!texBump.isEmpty())
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