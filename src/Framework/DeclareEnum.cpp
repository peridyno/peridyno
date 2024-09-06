#include "DeclareEnum.h"
#include <vector>
#include <string>

namespace dyno 
{
	std::string earse_spacing(const std::string& str)
	{
		if (str.empty())
			return std::string("");
		std::string tmpStr;
		for (int i = 0; i < str.length(); i++)
			if (str[i] != ' ')
				tmpStr += str[i];

		return tmpStr;
	}


	bool parse_enum_string(const std::string& enum_str, EnumMap& enumKeyValueList)
	{
		std::vector<std::string> enum_value_list;

		size_t npos = enum_str.find(",");
		size_t nlastpos = 0;
		while (npos != std::string::npos)
		{
			enum_value_list.push_back(enum_str.substr(nlastpos, npos - nlastpos));
			nlastpos = npos + 1;
			npos = enum_str.find(",", static_cast<unsigned int>(nlastpos));
		}
		if (nlastpos != enum_str.length())
		{
			enum_value_list.push_back(enum_str.substr(nlastpos, enum_str.length() - nlastpos));
		}
		if (enum_value_list.size() == 0)
			return false;

		enumKeyValueList.clear();
		int nDefaultValue = 0;
		for (std::vector<std::string>::iterator itor = enum_value_list.begin(); itor != enum_value_list.end(); itor++)
		{
			std::string str_enum_field = earse_spacing(*itor);
			long nEnumValue;
			std::string str_enum_field_name;

			int nPos = str_enum_field.find("=");
			if (nPos != std::string::npos)
			{
				char tmpKeyValue[64] = { '\0' };
				std::string tmpValue_;
				str_enum_field_name = str_enum_field.substr(0, nPos);

				tmpValue_ = str_enum_field.substr(nPos + 1, (*itor).length());
				sscanf(tmpValue_.c_str(), "%[^LlUu]", tmpKeyValue);
				tmpValue_ = tmpKeyValue;
				if (tmpValue_.find("0x") != std::string::npos)
					nEnumValue = strtol(tmpKeyValue, NULL, 16);
				else if (tmpValue_[0] == '0')
					nEnumValue = strtol(tmpKeyValue, NULL, 8);
				else
					nEnumValue = strtol(tmpKeyValue, NULL, 10);
			}
			else
			{
				str_enum_field_name = str_enum_field;
				nEnumValue = nDefaultValue;
			}
			nDefaultValue = nEnumValue + 1;

			enumKeyValueList[nEnumValue] = str_enum_field_name;
		}

		enum_value_list.clear();

		if (enumKeyValueList.size() == 0)
			return false;


		return true;
	}

	void PEnum::setCurrentKey(int index)
	{
		if (m_enum_map.find(index) == m_enum_map.end())
			return;

		m_enum_value = index;
		m_enum_name = m_enum_map[index];
	}
}
