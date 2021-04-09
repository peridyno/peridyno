#pragma once
#include "Array/Array.h"
#include "Framework/FieldBase.h"
#include "Framework/Base.h"
#include "Topology/NeighborList.h"

namespace dyno {

template<typename T>
class NeighborField : public FieldBase
{
public:
	typedef T VarType;
	typedef NeighborList<T> DataType;

	NeighborField();
	NeighborField(int num, int nbrSize = 0);
	NeighborField(std::string name, std::string description, int num = 1, int nbrSize = 0);
	NeighborField(std::string name, std::string description, FieldTypeEnum fieldType, Base* parent, int num, int nbrSize);
	~NeighborField() override;

	uint getElementCount() override {
		auto ref = this->getDataPtr();
		return ref == nullptr ? 0 : ref->size();;
	}
	void setElementCount(int num, int nbrSize = 0);
//	void resize(int num);
	const std::string getTemplateName() override { return std::string(typeid(T).name()); }
	const std::string getClassName() override { return std::string("NeighborField"); }
//	DeviceType getDeviceType() override { return DeviceType::GPU; }

	std::shared_ptr<NeighborList<T>> getDataPtr();

	NeighborList<T>& getData() { return *getDataPtr(); }

	bool isEmpty() override {
		return getDataPtr() == nullptr;
	}

	bool connect(NeighborField<T>* field2);

	NeighborField<T>* getSourceNeighborField();

private:
	std::shared_ptr<NeighborList<T>> m_data = nullptr;
};

template<typename T>
NeighborField<T>::NeighborField()
	:FieldBase("", "")
	, m_data(nullptr)
{
}


template<typename T>
NeighborField<T>::NeighborField(int num, int nbrSize /*= 0*/)
	:FieldBase("", "")
{
	m_data = std::make_shared<NeighborList<T>>();
	m_data->resize(num);
	if (nbrSize != 0)
	{
		m_data->setNeighborLimit(nbrSize);
	}
	else
	{
		m_data->setDynamic();
	}
}

template<typename T>
void NeighborField<T>::setElementCount(int num, int nbrSize /*= 0*/)
{
	auto arr = this->getSourceNeighborField();
	//std::shared_ptr<NeighborList<T>> data = getDataPtr();
	if (arr == nullptr)
	{
		m_data = num <= 0 ? nullptr : std::make_shared<NeighborList<T>>(num, nbrSize);
	}
	else
	{
		if(arr->m_data != nullptr)
			arr->m_data->release();
		arr->m_data = num <= 0 ? nullptr : std::make_shared<NeighborList<T>>(num, nbrSize);
	}
}

template<typename T>
NeighborField<T>::NeighborField(std::string name, std::string description, int num, int nbrSize)
	: FieldBase(name, description)
{
	m_data = std::make_shared<NeighborList<T>>();
	m_data->resize(num);
	if (nbrSize != 0)
	{
		m_data->setNeighborLimit(nbrSize);
	}
	else
	{
		m_data->setDynamic();
	}
}


template<typename T>
NeighborField<T>::NeighborField(std::string name, std::string description, FieldTypeEnum fieldType, Base* parent, int num, int nbrSize)
	: FieldBase(name, description, fieldType, parent)
{
	m_data = num <= 0 ? nullptr : std::make_shared<NeighborList<T>>(num, nbrSize);
}


// template<typename T>
// void NeighborField<T>::resize(int num)
// {
// 	m_data->resize(num);
// }

template<typename T>
NeighborField<T>::~NeighborField()
{
	if (m_data.use_count() == 1)
	{
		m_data->release();
	}
}

template<typename T>
bool NeighborField<T>::connect(NeighborField<T>* field2)
{
	this->connectField(field2);
	return true;
}

template<typename T>
std::shared_ptr<NeighborList<T>> NeighborField<T>::getDataPtr()
{
	FieldBase* source = getSource();
	if (source == nullptr)
	{
		return m_data;
	}
	else
	{
		NeighborField<T>* var = dynamic_cast<NeighborField<T>*>(source);
		if (var != nullptr)
		{
			//return var->getDataPtr();
			return (*var).getDataPtr();
		}
		else
		{
			return nullptr;
		}
	}
}

template<typename T>
NeighborField<T>* NeighborField<T>::getSourceNeighborField()
{
	FieldBase* source = getSource();
	if (source == nullptr)
	{
		return this;
	}
	else
	{
		NeighborField<T>* var = dynamic_cast<NeighborField<T>*>(source);
		if (var == nullptr)
			return nullptr;
		else
			return var->getSourceNeighborField();
	}
}

}
