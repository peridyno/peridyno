#include "DataSource.h"

namespace dyno 
{
	bool DataSource::captionVisible()
	{
		return false;
	}

	IMPLEMENT_TCLASS(FloatingNumber, TDataType);

	template<typename TDataType>
	FloatingNumber<TDataType>::FloatingNumber()
	{
		this->outFloating()->setValue(this->varValue()->getData());

		auto callback = std::make_shared<FCallBackFunc>(
			[=]() {this->outFloating()->setValue(this->varValue()->getData()); }
		);

		this->varValue()->attach(callback);
	}

	template class FloatingNumber<DataType3f>;
	template class FloatingNumber<DataType3d>;

	IMPLEMENT_TCLASS(Vector3Source, TDataType);

	template<typename TDataType>
	Vector3Source<TDataType>::Vector3Source()
	{
		this->outVector()->setValue(this->varValue()->getData());

		auto callback = std::make_shared<FCallBackFunc>(
			[=]() {this->outVector()->setValue(this->varValue()->getData()); }
		);

		this->varValue()->attach(callback);
	}

	template class Vector3Source<DataType3f>;
	template class Vector3Source<DataType3d>;
}