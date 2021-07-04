#include "Base.h"
#include <algorithm>
#include "Framework/FBase.h"
#include "Framework/Log.h"
using std::find;

namespace dyno {

bool Base::addField(FBase* data)
{
	return addField(data->getObjectName(), data);
}

bool Base::addField(FieldID name, FBase* data)
{
	if (findField(data) == NULL)
	{
		m_field.push_back(data);
	}
	else
	{
		std::cout << "Data field " << name
			<< " already exists in this class !"
			<< std::endl;
		return false;
	}

	addFieldAlias(name, data);

	return true;
}

bool Base::addFieldAlias(FieldID name, FBase* data)
{
	if (findFieldAlias(name) == NULL)
	{
		m_fieldAlias.insert(std::make_pair(name, data));
	}
	else
	{
		if (data != getField(name))
		{
			std::cout << "Field name " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return false;
		}

	}

}

bool Base::addFieldAlias(FieldID name, FBase* data, FieldMap& fieldAlias)
{
	if (findFieldAlias(name, fieldAlias) == NULL)
	{
		fieldAlias.insert(std::make_pair(name, data));
	}
	else
	{
		if (data != getField(name))
		{
			std::cout << "Field name " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return false;
		}

	}
}

bool Base::findField(FBase* data)
{
	FieldVector::iterator result = find(m_field.begin(), m_field.end(), data);
	// return false if no field is found!
	if (result == m_field.end())
	{
		return false;
	}
	return true;
}

bool Base::findFieldAlias(const FieldID name)
{
	FieldMap::iterator result = m_fieldAlias.find(name);
	// return false if no alias is found!
	if (result == m_fieldAlias.end())
	{
		return false;
	}
	return true;
}

bool Base::findFieldAlias(const FieldID name, FieldMap& fieldAlias)
{
	FieldMap::iterator result = fieldAlias.find(name);
	// return false if no alias is found!
	if (result == fieldAlias.end())
	{
		return false;
	}
	return true;
}

bool Base::removeField(FBase* data)
{
	FieldVector::iterator result = find(m_field.begin(), m_field.end(), data);
	if (result == m_field.end())
	{
		return false;
	}

	m_field.erase(result);

	FieldMap::iterator iter;
	for (iter = m_fieldAlias.begin(); iter != m_fieldAlias.end();)
	{
		if (iter->second == data)
		{
			m_fieldAlias.erase(iter++);
		}
		else
		{
			++iter;
		}
	}

	return true;
}

bool Base::removeFieldAlias(const FieldID name)
{
	return removeFieldAlias(name, m_fieldAlias);
}

bool Base::removeFieldAlias(const FieldID name, FieldMap& fieldAlias)
{
	FieldMap::iterator iter = fieldAlias.find(name);
	if (iter != fieldAlias.end())
	{
		FBase* data = iter->second;

		fieldAlias.erase(iter);

		if (getFieldAliasCount(data) == 0)
		{
			removeField(data);
		}
		return true;
	}

	return false;
}

FBase* Base::getField(const FieldID name)
{
	FieldMap::iterator iter = m_fieldAlias.find(name);
	if (iter != m_fieldAlias.end())
	{
		return iter->second;
	}
	return nullptr;
}

std::vector<FBase*>& Base::getAllFields()
{
	return m_field;
}

bool Base::isAllFieldsReady()
{
	bool bReady = true;
	for (int i = 0; i < m_field.size(); i++)
	{
		bReady = bReady & !m_field[i]->isEmpty();
		if (!bReady)
		{
			break;
		}
	}
	return bReady;
}


std::vector<std::string> Base::getFieldAlias(FBase* field)
{
	std::vector<FieldID> names;
	FieldMap::iterator iter;
	for (iter = m_fieldAlias.begin(); iter != m_fieldAlias.end(); iter++)
	{
		if (iter->second == field)
		{
			names.push_back(iter->first);
		}
	}
	return names;
}

int Base::getFieldAliasCount(FBase* data)
{
	int num = 0;
	FieldMap::iterator iter;
	for (iter = m_fieldAlias.begin(); iter != m_fieldAlias.end(); iter++)
	{
		if (iter->second == data)
		{
			num++;
		}
	}
	return num;
}

bool Base::attachField(FBase* field, std::string name, std::string desc, bool autoDestroy /*= true*/)
{
	field->setParent(this);
	field->setObjectName(name);
	field->setDescription(desc);
	field->setAutoDestroy(autoDestroy);
	bool ret = addField(field);

	if (!ret)
	{
		Log::sendMessage(Log::Error, std::string("The field ") + name + std::string(" already exists!"));
	}
	return ret;
}
}