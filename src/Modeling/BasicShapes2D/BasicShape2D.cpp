#include "BasicShape2D.h"

namespace dyno
{
	template<typename TDataType>
	BasicShape2D<TDataType>::BasicShape2D()
		: ParametricModel<TDataType>()
	{
		this->varLocation()->setActive(false);
		this->varRotation()->setActive(false);
		this->varScale()->setActive(false);

		this->varLocation()->attach(std::make_shared<FCallBackFunc>(
			[=]() {
				auto loc = this->varLocation()->getValue();
				this->varLocation2D()->setValue(Coord2D(loc[0], loc[1]));
			}
		));

		this->varRotation()->attach(std::make_shared<FCallBackFunc>(
			[=]() {
				auto rot = this->varRotation()->getValue();
				this->varRotation2D()->setValue(rot[2]);
			}
		));

		this->varScale()->attach(std::make_shared<FCallBackFunc>(
			[=]() {
				auto sc = this->varScale()->getValue();
				this->varScale2D()->setValue(Coord2D(sc[0], sc[1]));
			}
		));
	}

	DEFINE_CLASS(BasicShape2D);
}