#include "ModuleFields.h"
#include "Framework/Log.h"
namespace dyno
{
	ModuleFields::ModuleFields(std::string name)
		: Module(name)
	{
		std::function<void()> callback = std::bind(&ModuleFields::calculateRectangleArea, this);

		this->inWidth()->setCallBackFunc(callback);
		this->inHeight()->setCallBackFunc(callback);
	}

	CALLBACK ModuleFields::calculateRectangleArea()
	{
		if (this->inWidth()->isEmpty() || this->inHeight()->isEmpty())
		{
			Log::sendMessage(Log::Warning, "Either the width or height of the rectangle is not set!");
			return;
		}

		this->outArea()->setValue(this->inWidth()->getData()*this->inHeight()->getData());
	}
}