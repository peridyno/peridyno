#include "WSampleWidget.h"

#include <Wt/WPushButton.h>
#include <Wt/WGridLayout.h>
#include <Wt/WVBoxLayout.h>
#include <Wt/WImage.h>
#include <Wt/WLabel.h>
#include <Wt/WCssDecorationStyle.h>


#define DECLARE_PHYSIKA_SAMPLE(NAME) \
	class NAME : public Sample { \
	public:\
		std::string name() const {return #NAME;};\
		std::string thumbnail() const {return std::string("samples/") + #NAME + ".jpg";};\
		std::string source() const {return std::string("/") + #NAME + ".py";};\
	};

DECLARE_PHYSIKA_SAMPLE(Collision)
DECLARE_PHYSIKA_SAMPLE(DrySand)
DECLARE_PHYSIKA_SAMPLE(Elasticity)
DECLARE_PHYSIKA_SAMPLE(Fracture)
DECLARE_PHYSIKA_SAMPLE(HyperElasticity)
DECLARE_PHYSIKA_SAMPLE(MultiFluid)
DECLARE_PHYSIKA_SAMPLE(Plasticity)
DECLARE_PHYSIKA_SAMPLE(Rod)
DECLARE_PHYSIKA_SAMPLE(SFI)
DECLARE_PHYSIKA_SAMPLE(SingleFluid)
DECLARE_PHYSIKA_SAMPLE(SWE)
DECLARE_PHYSIKA_SAMPLE(ViscoPlasticity)
DECLARE_PHYSIKA_SAMPLE(WetSand)

class SampleStore
{
private:
	SampleStore()
	{
		samples.push_back(new Collision);
		samples.push_back(new DrySand);
		samples.push_back(new Elasticity);
		samples.push_back(new Fracture);
		samples.push_back(new HyperElasticity);
		samples.push_back(new MultiFluid);
		samples.push_back(new Plasticity);
		samples.push_back(new Rod);
		samples.push_back(new SFI);
		samples.push_back(new SingleFluid);
		samples.push_back(new SWE);
		samples.push_back(new ViscoPlasticity);
		samples.push_back(new WetSand);
	}

public:
	static SampleStore* getInstance()
	{
		static SampleStore* instance = new SampleStore;
		return instance;
	}

	std::vector<Sample*>& getSamples()
	{
		return samples;
	}

	Sample* getSample(int idx)
	{
		return samples[idx];
	}

	void addSample(Sample* sample)
	{
		samples.push_back(sample);
	}

private:
	std::vector<Sample*> samples;

};

Wt::WContainerWidget* _createItem(Sample* sample)
{
	Wt::WContainerWidget* container = new Wt::WContainerWidget;
	auto layout = container->setLayout(std::make_unique<Wt::WVBoxLayout>());

	// image
	auto image = layout->addWidget(std::make_unique<Wt::WImage>(sample->thumbnail()), 0, Wt::AlignmentFlag::Center);
	image->resize(120, 120);
	// label
	auto label = layout->addWidget(std::make_unique<Wt::WLabel>(sample->name()), 0, Wt::AlignmentFlag::Center);
	// description as tooltip
	container->setToolTip(sample->description());

	Wt::WCssDecorationStyle style0 = container->decorationStyle();
	Wt::WCssDecorationStyle style1 = container->decorationStyle();
	style0.setBackgroundColor(Wt::WColor(200, 200, 200));
	style0.setForegroundColor(Wt::WColor(50, 50, 50));

	container->setStyleClass("sample-item");
	image->setStyleClass("sample-item");

	container->mouseWentOver().connect([=]() {
		container->setDecorationStyle(style0);

		});

	container->mouseWentOut().connect([=]() {
		container->setDecorationStyle(style1);
		});

	return container;
}

WSampleWidget::WSampleWidget()
{
	this->setHeight(Wt::WLength("100%"));
	this->setOverflow(Wt::Overflow::Auto);

	for (auto sample : SampleStore::getInstance()->getSamples())
	{
		auto item = _createItem(sample);
		this->addWidget(std::unique_ptr<Wt::WContainerWidget>(item));

		item->clicked().connect([=]()
			{
				m_signal.emit(sample);
			});
	}
}


Wt::Signal<Sample*>& WSampleWidget::clicked()
{
	return m_signal;
}
