#include "CarMaterial.h"


namespace dyno
{
	CustomCarMaterial::CustomCarMaterial() :CustomMaterial()
	{
		mCarMaterial = std::make_shared<CarMaterial>();
		initialVar();
		this->outMaterial()->setDataPtr(mCarMaterial);
	}

	CustomCarMaterial::CustomCarMaterial(const std::string& name) :CustomMaterial(name)
	{
		mCarMaterial = std::make_shared<CarMaterial>();
		initialVar();
		this->outMaterial()->setDataPtr(mCarMaterial);
	}

	CustomCarMaterial::CustomCarMaterial(const std::shared_ptr<MaterialLoaderModule>& MaterialLoaderPtr, std::shared_ptr<BreakMaterial>& BreakMaterialModule, std::string Name)
		: CustomMaterial(MaterialLoaderPtr, BreakMaterialModule, Name) 
	{
		mCarMaterial = std::make_shared<CarMaterial>();
		initialVar();
		this->outMaterial()->setDataPtr(mCarMaterial);
	}

	void CustomCarMaterial::updateImpl()
	{
		if (!this->inTexLightMask()->isEmpty() && mCarMaterial)
		{
			mCarMaterial->texLightMask.assign(this->inTexLightMask()->getData());
		}

		CustomMaterial::updateImpl();
	}

	void CustomCarMaterial::initialVar()
	{
		CustomMaterial::initialVar();
		this->inTexLightMask()->tagOptional(true);
	}

	IMPLEMENT_CLASS(CustomCarMaterial);
	GLCarMaterial::GLCarMaterial()
	{
	}
	GLCarMaterial::~GLCarMaterial()
	{
		release();
	}
	void GLCarMaterial::create()
	{
	}
	void GLCarMaterial::release()
	{
		GLMaterial::release();
		texLightMask.release();

	}
	void GLCarMaterial::updateGL()
	{
		GLMaterial::updateGL();
		texLightMask.updateGL();

	}
}