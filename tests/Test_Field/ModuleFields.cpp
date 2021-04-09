#include "ModuleFields.h"
#include "Framework/Log.h"
namespace dyno
{
	ModuleFields::ModuleFields(std::string name)
		: Module(name)
	{
		std::function<void()> callback = std::bind(&ModuleFields::calculateRectangleArea, this);

		this->varWidth()->setCallBackFunc(callback);
		this->varHeight()->setCallBackFunc(callback);
	}

	CALLBACK ModuleFields::calculateRectangleArea()
	{
		if (this->varWidth()->isEmpty() || this->varHeight()->isEmpty())
		{
			Log::sendMessage(Log::Warning, "Either the width or height of the rectangle is not set!");
			return;
		}

		this->varArea()->setValue(this->varWidth()->getData()*this->varHeight()->getData());
	}
}