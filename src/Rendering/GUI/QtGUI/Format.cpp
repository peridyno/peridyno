#include "Format.h"

#include <QRegularExpression>
#include <QVector>

namespace dyno
{
	QString FormatFieldWidgetName(std::string name)
	{
		QString qName = QString::fromStdString(name.c_str());

		bool isChinese = qName.contains(QRegularExpression("[\u4e00-\u9fa5]"));

		//If the string contains Chinese, show all the original string without splitting
		if (isChinese)
		{
			return qName;
		}

		//Otherwise, slit the name by the space
		QRegularExpression regexp("[A-Z][^A-Z]*");
		QRegularExpressionMatchIterator match = regexp.globalMatch(qName);
		QVector<QString> vec;

		QString ret;
		while (match.hasNext())
		{
			ret += match.next().captured() + " ";
		}

		return ret;
	}

	QString FormatBlockPortName(std::string name)
	{
		QString qName = QString::fromStdString(name.c_str());

		bool isChinese = qName.contains(QRegularExpression("[\u4e00-\u9fa5]"));

		//If the string contains Chinese, show all the original string without splitting
		if (isChinese)
		{
			return qName;
		}

		QRegularExpression regexp0("[A-Za-z()]*");
		QRegularExpressionMatchIterator match0 = regexp0.globalMatch(qName);

		QString subStr = match0.hasNext() ? match0.next().captured() : QString("Port");

		QRegularExpression regexp("[A-Z][^A-Z]*");
		QRegularExpressionMatchIterator match = regexp.globalMatch(subStr);

		QString ret;
		while (match.hasNext())
		{
			ret += match.next().captured() + " ";
		}

		return ret;
	}

	QString FormatBlockCaptionName(std::string name)
	{
		QString qName = QString::fromStdString(name.c_str());

		bool isChinese = qName.contains(QRegularExpression("[\u4e00-\u9fa5]"));

		//If the string contains Chinese, show all the original string without splitting
		if (isChinese)
		{
			return qName;
		}

		QRegularExpression regexp0("[A-Za-z0-9]*");
		QRegularExpressionMatchIterator match0 = regexp0.globalMatch(qName);

		QString subStr = match0.hasNext() ? match0.next().captured() : QString("Port");

		QRegularExpression regexp("[A-Za-z0-9]*");
		QRegularExpressionMatchIterator match = regexp.globalMatch(subStr);

		QString ret;
		while (match.hasNext())
		{
			ret += match.next().captured() + QString(" ");
		}

		return ret;
	}

	QString FormatDescription(std::string name)
	{
		QString desc = QString::fromStdString(name.c_str());

// 		bool isChinese = qName.contains(QRegExp("[\\x4e00-\\x9fa5]+"));
// 
// 		//If the string contains Chinese, show all the original string without splitting
// 		if (isChinese)
// 		{
// 			return qName;
// 		}

		return desc;
	}

}
