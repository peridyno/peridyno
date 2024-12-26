#include "WLogWidget.h"

WLogWidget::WLogWidget()
{
	this->setLayoutSizeAware(true);
	this->setOverflow(Wt::Overflow::Auto);
	this->setHeight(Wt::WLength("100%"));
	this->setMargin(10);


}

WLogWidget::~WLogWidget() { }