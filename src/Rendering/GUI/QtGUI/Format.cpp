#include "Format.h"

#include <QRegularExpression>
#include <QVector>
#include <QTextCodec>

namespace dyno
{
	QString FormatFieldWidgetName(std::string name)
	{
		QTextCodec* codec = QTextCodec::codecForName("GB2312");

		QString qName = codec->toUnicode(name.c_str());

		bool isChinese = qName.contains(QRegExp("[\\x4e00-\\x9fa5]+"));

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
		QTextCodec* codec = QTextCodec::codecForName("GB2312");

		QString qName = codec->toUnicode(name.c_str());

		bool isChinese = qName.contains(QRegExp("[\\x4e00-\\x9fa5]+"));

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
		QTextCodec* codec = QTextCodec::codecForName("GB2312");

		QString qName = codec->toUnicode(name.c_str());

		bool isChinese = qName.contains(QRegExp("[\\x4e00-\\x9fa5]+"));

		//If the string contains Chinese, show all the original string without splitting
		if (isChinese)
		{
			return qName;
		}

		QRegularExpression regexp0("[A-Za-z()\\s]*");
		QRegularExpressionMatchIterator match0 = regexp0.globalMatch(qName);

		QString subStr = match0.hasNext() ? match0.next().captured() : QString("Port");

		QRegularExpression regexp("[A-Za-z()\\s]*");
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
		QTextCodec* codec = QTextCodec::codecForName("GB2312");

		QString desc = codec->toUnicode(name.c_str());

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
