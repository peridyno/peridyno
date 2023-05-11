#include "DataSource.h"

namespace dyno 
{
	bool DataSource::captionVisible()
	{
		return false;
	}

	IMPLEMENT_CLASS(FloatSource);

	FloatSource::FloatSource()
	{
		this->outFloat()->setValue(this->varValue()->getData());

		auto callback = std::make_shared<FCallBackFunc>(
			[=]() {this->outFloat()->setValue(this->varValue()->getData()); }
		);

		this->varValue()->attach(callback);
	}

	IMPLEMENT_CLASS(DoubleSource);

	DoubleSource::DoubleSource()
	{
		this->outDouble()->setValue(this->varValue()->getData());

		auto callback = std::make_shared<FCallBackFunc>(
			[=]() {this->outDouble()->setValue(this->varValue()->getData()); }
		);

		this->varValue()->attach(callback);
	}

	IMPLEMENT_CLASS(Vec3fSource);

	Vec3fSource::Vec3fSource()
	{
		this->outVector3f()->setValue(this->varValue()->getData());

		auto callback = std::make_shared<FCallBackFunc>(
			[=]() {this->outVector3f()->setValue(this->varValue()->getData()); }
		);

		this->varValue()->attach(callback);
	}

	IMPLEMENT_CLASS(Vec3dSource);

	Vec3dSource::Vec3dSource()
	{
		this->outVector3d()->setValue(this->varValue()->getData());

		auto callback = std::make_shared<FCallBackFunc>(
			[=]() {this->outVector3d()->setValue(this->varValue()->getData()); }
		);

		this->varValue()->attach(callback);
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