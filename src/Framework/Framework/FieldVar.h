#pragma once
#include <iostream>
#include "Typedef.h"
#include "Field.h"
#include "Base.h"
#include "Framework/Log.h"

namespace dyno {

/*!
*	\class	Variable
*	\brief	Variables of build-in data types.
*/
template<typename T>
class VarField : public Field
{
public:
	typedef T VarType;
	typedef T DataType;
	typedef VarField<T> FieldClass;

	DEFINE_DERIVED_FUNC(FieldClass, DataType);

	VarField();
	VarField(std::string name, std::string description, EFieldType fieldType, Base* parent);
	VarField(T value, std::string name, std::string description, EFieldType fieldType, Base* parent);
	~VarField() override;

	size_t getElementCount() override { return 1; }
	const std::string getTemplateName() override { return std::string(typeid(T).name()); }
	const std::string getClassName() override { return std::string("Variable"); }

	void setValue(T val);
};

template<typename T>
VarField<T>::VarField()
	: Field("", "")
{
}

template<typename T>
VarField<T>::VarField(std::string name, std::string description, EFieldType fieldType, Base* parent)
	: Field(name, description, fieldType, parent)
	, m_data(nullptr)
{
}

template<typename T>
VarField<T>::VarField(T value, std::string name, std::string description, EFieldType fieldType, Base* parent)
	: Field(name, description, fieldType, parent)
{
	this->setValue(value);
}

template<typename T>
VarField<T>::~VarField()
{
};

template<typename T>
void VarField<T>::setValue(T val)
{
	std::shared_ptr<T>& data = this->getReference();
	if (data == nullptr)
	{
		data = std::make_shared<T>(val);
	}
	else
	{
		*data = val;
	}

	this->update();
}

template<typename T>
using HostVarField = VarField<T>;

template<typename T>
using DeviceVarField = VarField<T>;

template<typename T>
using HostVariablePtr = std::shared_ptr< HostVarField<T> >;

template<typename T>
using DeviceVariablePtr = std::shared_ptr< DeviceVarField<T> >;
}