#include "WLogWidget.h"

WLogMessage* WLogMessage::instance = nullptr;

WLogMessage::WLogMessage()
{
	dyno::Log::setUserReceiver(RecieveLogMessage);

	WLogMessage::instance = this;

}

WLogMessage::~WLogMessage()
{
	WLogMessage::instance = nullptr;
}

void WLogMessage::RecieveLogMessage(const dyno::Log::Message& m)
{
	if (WLogMessage::instance) {
		WLogMessage::instance->updateLog(m.text.c_str());
	}
}

void WLogMessage::updateLog(const char* text)
{
	message = message + text + "\n";
	m_signal.emit(message);
}


Wt::Signal<std::string>& WLogMessage::updateText()
{
	return m_signal;
}

WLogWidget::WLogWidget(WMainWindow* parent)
	: mParent(parent)
{
	this->setLayoutSizeAware(true);
	//this->setOverflow(Wt::Overflow::Auto);
	this->setMargin(10);

	text = this->addNew<Wt::WTextArea>();
	//text->setHeight(Wt::WLength("95%"));
	text->resize("95%", "95%");
	text->setStyleClass("save-middle");

	auto downloadButton = this->addWidget(std::make_unique< Wt::WPushButton>("Update"));
	downloadButton->setMargin(10, Wt::Side::Top);
	downloadButton->setStyleClass("btn-primary");

	downloadButton->clicked().connect([=]
		{
			auto mScene = parent->getScene();
			std::ostringstream oss;
			oss << mScene;
			std::string filePath = oss.str() + ".txt";
			std::string content;

			std::ifstream fileStream(filePath);
			if (!fileStream.is_open()) {
				std::cerr << "Unable to open file for reading." << std::endl;
				return 1;
			}

			std::getline(fileStream, content, '\0'); // 读取整个文件内容
			fileStream.close();

			text->setText(content);

			if (std::remove(filePath.c_str()) == 0) {
				std::cout << "File successfully deleted." << std::endl;
			}
			else {
				std::cerr << "Error deleting file." << std::endl;
			}

		});
}

WLogWidget::~WLogWidget() {}

void WLogWidget::showMessage(std::string s)
{
	text->setText("123");
}
