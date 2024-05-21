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
		std::string source() const {return std::string("../../../../data/python_web_sample/") + #NAME + ".py";};\
	};

DECLARE_PHYSIKA_SAMPLE(CPD_ClothDrop)
DECLARE_PHYSIKA_SAMPLE(CPD_ClothOnTable)
DECLARE_PHYSIKA_SAMPLE(CPD_ClothOverBall_1)
DECLARE_PHYSIKA_SAMPLE(CPD_ClothOverBall_3)
DECLARE_PHYSIKA_SAMPLE(CPD_ClothOverBall_6)
DECLARE_PHYSIKA_SAMPLE(CPD_RealTimeCloth_v1)
DECLARE_PHYSIKA_SAMPLE(CPD_RealTimeCloth_v2)
DECLARE_PHYSIKA_SAMPLE(GL_ClothWithCollision)

DECLARE_PHYSIKA_SAMPLE(DualParticle_4Box)
DECLARE_PHYSIKA_SAMPLE(DualParticle_CollidingFish)
DECLARE_PHYSIKA_SAMPLE(DualParticle_Fountain)
DECLARE_PHYSIKA_SAMPLE(DualParticle_Jets)

DECLARE_PHYSIKA_SAMPLE(GL_CapillaryWave)
DECLARE_PHYSIKA_SAMPLE(GL_GranularMedia)
DECLARE_PHYSIKA_SAMPLE(GL_Ocean)
DECLARE_PHYSIKA_SAMPLE(GL_OceanPatch)
DECLARE_PHYSIKA_SAMPLE(GL_Terrain)

DECLARE_PHYSIKA_SAMPLE(Qt_GLTF)

DECLARE_PHYSIKA_SAMPLE(GL_Cloth)
DECLARE_PHYSIKA_SAMPLE(GL_Elasticity)
DECLARE_PHYSIKA_SAMPLE(GL_Plasticity_Error)

DECLARE_PHYSIKA_SAMPLE(GL_BoxStack)
DECLARE_PHYSIKA_SAMPLE(GL_Bricks)
DECLARE_PHYSIKA_SAMPLE(GL_CollisionDetectionIn3D)
DECLARE_PHYSIKA_SAMPLE(GL_CollisionMask)
DECLARE_PHYSIKA_SAMPLE(GL_FixedJoint)

DECLARE_PHYSIKA_SAMPLE(GL_Semi_Barricade)
DECLARE_PHYSIKA_SAMPLE(Qt_WaterPouring)

DECLARE_PHYSIKA_SAMPLE(GL_GhostSPH_Error)
DECLARE_PHYSIKA_SAMPLE(GL_ParticleEmitter)
DECLARE_PHYSIKA_SAMPLE(GL_ParticleFluid)
DECLARE_PHYSIKA_SAMPLE(GL_SemiImplicitDensitySolver)
DECLARE_PHYSIKA_SAMPLE(GL_ViscosityFish)

DECLARE_PHYSIKA_SAMPLE(GL_GlfwGUI)

//DECLARE_PHYSIKA_SAMPLE(DrySand)
//DECLARE_PHYSIKA_SAMPLE(Elasticity)
//DECLARE_PHYSIKA_SAMPLE(Fracture)
//DECLARE_PHYSIKA_SAMPLE(HyperElasticity)
//DECLARE_PHYSIKA_SAMPLE(MultiFluid)
//DECLARE_PHYSIKA_SAMPLE(Plasticity)
//DECLARE_PHYSIKA_SAMPLE(Rod)
//DECLARE_PHYSIKA_SAMPLE(SFI)
//DECLARE_PHYSIKA_SAMPLE(SingleFluid)
//DECLARE_PHYSIKA_SAMPLE(SWE)
//DECLARE_PHYSIKA_SAMPLE(ViscoPlasticity)
//DECLARE_PHYSIKA_SAMPLE(WetSand)

class SampleStore
{
private:
	SampleStore()
	{
		samples.push_back(new CPD_ClothDrop);
		samples.push_back(new CPD_ClothOnTable);
		samples.push_back(new CPD_ClothOverBall_1);
		samples.push_back(new CPD_ClothOverBall_3);
		samples.push_back(new CPD_ClothOverBall_6);
		samples.push_back(new CPD_RealTimeCloth_v1);
		samples.push_back(new CPD_RealTimeCloth_v2);
		samples.push_back(new GL_ClothWithCollision);

		samples.push_back(new DualParticle_4Box);
		samples.push_back(new DualParticle_CollidingFish);
		samples.push_back(new DualParticle_Fountain);
		samples.push_back(new DualParticle_Jets);

		samples.push_back(new GL_CapillaryWave);
		samples.push_back(new GL_GranularMedia);
		samples.push_back(new GL_Ocean);
		samples.push_back(new GL_OceanPatch);
		samples.push_back(new GL_Terrain);

		samples.push_back(new Qt_GLTF);

		samples.push_back(new GL_Cloth);
		samples.push_back(new GL_Elasticity);
		samples.push_back(new GL_Plasticity_Error);

		samples.push_back(new GL_BoxStack);
		samples.push_back(new GL_Bricks);
		samples.push_back(new GL_CollisionDetectionIn3D);
		samples.push_back(new GL_CollisionMask);
		samples.push_back(new GL_FixedJoint);

		samples.push_back(new GL_Semi_Barricade);
		samples.push_back(new Qt_WaterPouring);

		samples.push_back(new GL_GhostSPH_Error);
		samples.push_back(new GL_ParticleEmitter);
		samples.push_back(new GL_ParticleFluid);
		samples.push_back(new GL_SemiImplicitDensitySolver);
		samples.push_back(new GL_ViscosityFish);

		samples.push_back(new GL_GlfwGUI);
		//samples.push_back(new DrySand);
		//samples.push_back(new Elasticity);
		//samples.push_back(new Fracture);
		//samples.push_back(new HyperElasticity);
		//samples.push_back(new MultiFluid);
		//samples.push_back(new Plasticity);
		//samples.push_back(new Rod);
		//samples.push_back(new SFI);
		//samples.push_back(new SingleFluid);
		//samples.push_back(new SWE);
		//samples.push_back(new ViscoPlasticity);
		//samples.push_back(new WetSand);
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

	auto container = std::make_unique<Wt::WContainerWidget>();
	auto gridLayout = std::make_unique<Wt::WGridLayout>();

	int row = 0;
	int col = 0;

	for (auto sample : SampleStore::getInstance()->getSamples())
	{
		auto item = _createItem(sample);
		//this->addWidget(std::unique_ptr<Wt::WContainerWidget>(item));
		gridLayout->addWidget(std::unique_ptr<Wt::WContainerWidget>(item), row, col);

		item->clicked().connect([=]()
			{
				m_signal.emit(sample);
			});

		col++;
		if (col == 4)
		{
			col = 0;
			row++;
		}
	}
	container->setLayout(std::move(gridLayout));
	this->addWidget(std::move(container));
}

Wt::Signal<Sample*>& WSampleWidget::clicked()
{
	return m_signal;
}