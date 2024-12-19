#include "WSaveWidget.h"

WSaveWidget::WSaveWidget()
{
	this->setLayoutSizeAware(true);
	this->setOverflow(Wt::Overflow::Auto);
	this->setHeight(Wt::WLength("100%"));
	this->setMargin(0);

	auto layout = this->setLayout(std::make_unique<Wt::WVBoxLayout>());
	layout->setContentsMargins(0, 0, 0, 0);

	Wt::WPushButton* button = layout->addWidget(std::make_unique<Wt::WPushButton>("Save"));
	button->setStyleClass("btn-primary");

	button->clicked().connect(this, &WSaveWidget::save);
}

WSaveWidget::~WSaveWidget() {}

void WSaveWidget::save()
{
	auto scnLoader = dyno::SceneLoaderFactory::getInstance().getEntryByFileExtension("xml");
	scnLoader->save(dyno::SceneGraphFactory::instance()->active(), "save.xml");
}