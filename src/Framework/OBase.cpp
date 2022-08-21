#include "OBase.h"
#include <algorithm>
#include "FBase.h"
#include "Log.h"
using std::find;

namespace dyno {

	OBase::~OBase()
	{
		m_field.clear();
		m_fieldAlias.clear();

		fields_input.clear();
		fields_output.clear();
		fields_param.clear();
	}

	std::string OBase::caption()
	{
		return this->getClassInfo()->getClassName();
	}

	bool OBase::addField(FBase* data)
	{
		return addField(data->getObjectName(), data);
	}

	bool OBase::addField(FieldID name, FBase* data)
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

	bool OBase::addFieldAlias(FieldID name, FBase* data)
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

	bool OBase::addFieldAlias(FieldID name, FBase* data, FieldMap& fieldAlias)
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

	bool OBase::findField(FBase* data)
	{
		FieldVector::iterator result = find(m_field.begin(), m_field.end(), data);
		// return false if no field is found!
		if (result == m_field.end())
		{
			return false;
		}
		return true;
	}

	bool OBase::findFieldAlias(const FieldID name)
	{
		FieldMap::iterator result = m_fieldAlias.find(name);
		// return false if no alias is found!
		if (result == m_fieldAlias.end())
		{
			return false;
		}
		return true;
	}

	bool OBase::findFieldAlias(const FieldID name, FieldMap& fieldAlias)
	{
		FieldMap::iterator result = fieldAlias.find(name);
		// return false if no alias is found!
		if (result == fieldAlias.end())
		{
			return false;
		}
		return true;
	}

	bool OBase::removeField(FBase* data)
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

	bool OBase::removeFieldAlias(const FieldID name)
	{
		return removeFieldAlias(name, m_fieldAlias);
	}

	bool OBase::removeFieldAlias(const FieldID name, FieldMap& fieldAlias)
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

	FBase* OBase::getField(const FieldID name)
	{
		FieldMap::iterator iter = m_fieldAlias.find(name);
		if (iter != m_fieldAlias.end())
		{
			return iter->second;
		}
		return nullptr;
	}

	std::vector<FBase*>& OBase::getAllFields()
	{
		return m_field;
	}

	bool OBase::isAllFieldsReady()
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


	std::vector<std::string> OBase::getFieldAlias(FBase* field)
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

	int OBase::getFieldAliasCount(FBase* data)
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

	bool OBase::attachField(FBase* field, std::string name, std::string desc, bool autoDestroy /*= true*/)
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

	bool OBase::findInputField(FBase* field)
	{
		auto result = find(fields_input.begin(), fields_input.end(), field);
		// return false if no field is found!
		if (result == fields_input.end())
		{
			return false;
		}
		return true;
	}

	bool OBase::addInputField(FBase* field)
	{
		if (findInputField(field))
		{
			return false;
		}

		this->addField(field);

		fields_input.push_back(field);

		return true;
	}

	bool OBase::removeInputField(FBase* field)
	{
		if (!findInputField(field))
		{
			return false;
		}

		this->removeField(field);

		auto result = find(fields_input.begin(), fields_input.end(), field);
		if (result != fields_input.end())
		{
			fields_input.erase(result);
		}

		return true;
	}

	bool OBase::findOutputField(FBase* field)
	{
		auto result = find(fields_output.begin(), fields_output.end(), field);
		// return false if no field is found!
		if (result == fields_output.end())
		{
			return false;
		}
		return true;
	}

	bool OBase::addOutputField(FBase* field)
	{
		if (findOutputField(field))
		{
			return false;
		}

		this->addField(field);

		fields_output.push_back(field);

		return true;
	}

	bool OBase::addToOutput(FBase* field)
	{
		if (findOutputField(field))
		{
			return false;
		}

		fields_output.push_back(field);

		return true;
	}

	bool OBase::removeOutputField(FBase* field)
	{
		if (!findOutputField(field))
		{
			return false;
		}

		this->removeField(field);

		auto result = find(fields_output.begin(), fields_output.end(), field);
		if (result != fields_output.end())
		{
			fields_output.erase(result);
		}

		return true;
	}

	bool OBase::removeFromOutput(FBase* field)
	{
		if (!findOutputField(field))
		{
			return false;
		}

		auto result = find(fields_output.begin(), fields_output.end(), field);
		if (result != fields_output.end())
		{
			fields_output.erase(result);
		}

		return true;
	}

	bool OBase::findParameter(FBase* field)
	{
		auto result = find(fields_param.begin(), fields_param.end(), field);
		// return false if no field is found!
		if (result == fields_param.end())
		{
			return false;
		}
		return true;
	}

	bool OBase::addParameter(FBase* field)
	{
		if (findParameter(field))
		{
			return false;
		}

		this->addField(field);

		fields_param.push_back(field);

		return true;
	}

	bool OBase::removeParameter(FBase* field)
	{
		if (!findParameter(field))
		{
			return false;
		}

		this->removeField(field);

		auto result = find(fields_param.begin(), fields_param.end(), field);
		if (result != fields_output.end())
		{
			fields_param.erase(result);
		}

		return true;
	}

}