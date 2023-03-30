#include "CalculateArea.h"
#include "Log.h"
namespace dyno
{
	CalculateArea::CalculateArea(std::string name)
		: ComputeModule()
	{
	}

	void CalculateArea::compute()
	{
		float width = this->inWidth()->getData();
		float height = this->inHeight()->getData();

		this->outArea()->setValue(width*height);
	}
}