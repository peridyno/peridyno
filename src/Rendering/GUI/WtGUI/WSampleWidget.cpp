#include "WSampleWidget.h"

#include <Wt/WPushButton.h>
#include <Wt/WGridLayout.h>
#include <Wt/WVBoxLayout.h>
#include <Wt/WImage.h>
#include <Wt/WLabel.h>
#include <Wt/WCssDecorationStyle.h>

#include <Platform.h>

#define DECLARE_PERIDYNO_SAMPLE(NAME) \
	class NAME : public Sample { \
	public:\
		std::string name() const {return #NAME;};\
		std::string thumbnail() const {return std::string("samples/") + #NAME + ".jpg";};\
		std::string source() const {return getAssetPath() + std::string("python_web_sample/") + #NAME + ".py";};\
	};

// CodimensionalPD
class CPD_ClothDrop : public Sample {
public: std::string name() const {
	return "CPD_ClothDrop";
}; std::string thumbnail() const {
	return std::string("samples/") + "CPD_ClothDrop" + ".jpg";
}; std::string source() const {
	return getAssetPath() + std::string("python_web_sample/") + "CPD_ClothDrop" + ".py";
};
};
DECLARE_PERIDYNO_SAMPLE(CPD_ClothOnTable)
DECLARE_PERIDYNO_SAMPLE(CPD_ClothOverBall_1)
DECLARE_PERIDYNO_SAMPLE(CPD_ClothOverBall_3)
DECLARE_PERIDYNO_SAMPLE(CPD_ClothOverBall_6)
DECLARE_PERIDYNO_SAMPLE(CPD_RealTimeCloth_v1)
DECLARE_PERIDYNO_SAMPLE(CPD_RealTimeCloth_v2)
DECLARE_PERIDYNO_SAMPLE(CPD_RotateCylinder)
//DECLARE_PERIDYNO_SAMPLE(CPD_ShootingCloth)
DECLARE_PERIDYNO_SAMPLE(GL_ClothWithCollision)

// DualParticle
DECLARE_PERIDYNO_SAMPLE(DualParticle_4Box)
DECLARE_PERIDYNO_SAMPLE(DualParticle_CollidingFish)
DECLARE_PERIDYNO_SAMPLE(DualParticle_Fountain)
DECLARE_PERIDYNO_SAMPLE(DualParticle_Jets)
DECLARE_PERIDYNO_SAMPLE(Qt_DualParticle_FishAndBall)

// HeightField
DECLARE_PERIDYNO_SAMPLE(GL_CapillaryWave)
DECLARE_PERIDYNO_SAMPLE(GL_GranularMedia)
DECLARE_PERIDYNO_SAMPLE(GL_Ocean)
DECLARE_PERIDYNO_SAMPLE(GL_OceanPatch)
DECLARE_PERIDYNO_SAMPLE(GL_RigidSandCoupling)
DECLARE_PERIDYNO_SAMPLE(GL_Terrain)
DECLARE_PERIDYNO_SAMPLE(Qt_Buoyancy)
DECLARE_PERIDYNO_SAMPLE(Qt_LargeOcean)

// Modeling
DECLARE_PERIDYNO_SAMPLE(Qt_GLTF)
DECLARE_PERIDYNO_SAMPLE(Qt_GUI_ProcedureModeling)
//DECLARE_PERIDYNO_SAMPLE(Qt_Jeep)
DECLARE_PERIDYNO_SAMPLE(Qt_JeepLandscape)
//DECLARE_PERIDYNO_SAMPLE(Qt_PointSampling)
DECLARE_PERIDYNO_SAMPLE(Qt_PolyEdit)

// Peridynamics
//DECLARE_PERIDYNO_SAMPLE(GL_BrittleFracture)
DECLARE_PERIDYNO_SAMPLE(GL_Cloth)
DECLARE_PERIDYNO_SAMPLE(GL_Elasticity)
DECLARE_PERIDYNO_SAMPLE(GL_Plasticity)

// RigidBody
DECLARE_PERIDYNO_SAMPLE(GL_BoxStack)
DECLARE_PERIDYNO_SAMPLE(GL_Bricks)
//DECLARE_PERIDYNO_SAMPLE(GL_CollisionDetectionIn3D)
DECLARE_PERIDYNO_SAMPLE(GL_CollisionMask)
DECLARE_PERIDYNO_SAMPLE(GL_FixedJoint)
//DECLARE_PERIDYNO_SAMPLE(GL_HingeChain)
DECLARE_PERIDYNO_SAMPLE(GL_HingeJoint)
//DECLARE_PERIDYNO_SAMPLE(GL_HingeRing)
//DECLARE_PERIDYNO_SAMPLE(GL_Overlap)
//DECLARE_PERIDYNO_SAMPLE(GL_Pyramid)
//DECLARE_PERIDYNO_SAMPLE(GL_RigidCompound)
DECLARE_PERIDYNO_SAMPLE(GL_SliderJoint)
DECLARE_PERIDYNO_SAMPLE(GL_TestAttribute)
DECLARE_PERIDYNO_SAMPLE(GL_Timing)
DECLARE_PERIDYNO_SAMPLE(Qt_BallAndSocketJoint)
DECLARE_PERIDYNO_SAMPLE(Qt_Gear)
//DECLARE_PERIDYNO_SAMPLE(Qt_Multibody)
DECLARE_PERIDYNO_SAMPLE(Qt_PresetArticulatedBody)
DECLARE_PERIDYNO_SAMPLE(Qt_TrackedTank)

// SemiAnalytical
DECLARE_PERIDYNO_SAMPLE(GL_Semi_Barricade)
DECLARE_PERIDYNO_SAMPLE(Qt_WaterPouring)

// SPH
//DECLARE_PERIDYNO_SAMPLE(GL_Comparison)
//DECLARE_PERIDYNO_SAMPLE(GL_GhostSPH)
DECLARE_PERIDYNO_SAMPLE(GL_ParticleEmitter)
DECLARE_PERIDYNO_SAMPLE(GL_ParticleFluid)
DECLARE_PERIDYNO_SAMPLE(GL_ViscosityFish)
//DECLARE_PERIDYNO_SAMPLE(QT_GhostSPH)
DECLARE_PERIDYNO_SAMPLE(Qt_ParticleSkinning)

// Tutorials
DECLARE_PERIDYNO_SAMPLE(GL_GlfwGUI)
DECLARE_PERIDYNO_SAMPLE(GL_PhotorealisticRender)

// Volume
DECLARE_PERIDYNO_SAMPLE(GL_MarchingCubes)
DECLARE_PERIDYNO_SAMPLE(GL_SDFUniform)

class SampleStore
{
private:
	SampleStore()
	{
		// CodimensionalPD
		//samples.push_back(new CPD_ClothDrop);
		samples.push_back(new CPD_ClothOnTable);
		samples.push_back(new CPD_ClothOverBall_1);
		samples.push_back(new CPD_ClothOverBall_3);
		samples.push_back(new CPD_ClothOverBall_6);
		samples.push_back(new CPD_RealTimeCloth_v1);
		//samples.push_back(new CPD_RealTimeCloth_v2);
		//samples.push_back(new CPD_RotateCylinder);
		//samples.push_back(new CPD_ShootingCloth);
		samples.push_back(new GL_ClothWithCollision);

		// DualParticle
		samples.push_back(new DualParticle_4Box);
		samples.push_back(new DualParticle_CollidingFish);
		//samples.push_back(new DualParticle_Fountain);
		//samples.push_back(new DualParticle_Jets);
		samples.push_back(new Qt_DualParticle_FishAndBall);

		// HeightField
		samples.push_back(new GL_CapillaryWave);
		//samples.push_back(new GL_GranularMedia);
		samples.push_back(new GL_Ocean);
		samples.push_back(new GL_OceanPatch);
		samples.push_back(new GL_RigidSandCoupling);
		samples.push_back(new GL_Terrain);
		samples.push_back(new Qt_Buoyancy);
		samples.push_back(new Qt_LargeOcean);

		// Modeling
		samples.push_back(new Qt_GLTF);
		samples.push_back(new Qt_GUI_ProcedureModeling);
		//samples.push_back(new Qt_Jeep);
		samples.push_back(new Qt_JeepLandscape);
		//samples.push_back(new Qt_PointSampling);
		samples.push_back(new Qt_PolyEdit);

		// Peridynamics
		//samples.push_back(new GL_BrittleFracture);
		samples.push_back(new GL_Cloth);
		samples.push_back(new GL_Elasticity);
		samples.push_back(new GL_Plasticity);

		// RigidBody
		samples.push_back(new GL_BoxStack);
		samples.push_back(new GL_Bricks);
		//samples.push_back(new GL_CollisionDetectionIn3D);
		samples.push_back(new GL_CollisionMask);
		samples.push_back(new GL_FixedJoint);
		//samples.push_back(new GL_HingeChain);
		samples.push_back(new GL_HingeJoint);
		//samples.push_back(new GL_HingeRing);
		//samples.push_back(new GL_Overlap);
		//samples.push_back(new GL_Pyramid);
		//samples.push_back(new GL_RigidCompound);
		samples.push_back(new GL_SliderJoint);
		samples.push_back(new GL_TestAttribute);
		samples.push_back(new GL_Timing);
		samples.push_back(new Qt_BallAndSocketJoint);
		samples.push_back(new Qt_Gear);
		//samples.push_back(new Qt_Multibody);
		samples.push_back(new Qt_PresetArticulatedBody);
		samples.push_back(new Qt_TrackedTank);

		// SemiAnalytical
		samples.push_back(new GL_Semi_Barricade);
		samples.push_back(new Qt_WaterPouring);

		// SPH
		//samples.push_back(new GL_Comparison);
		//samples.push_back(new GL_GhostSPH);
		samples.push_back(new GL_ParticleEmitter);
		samples.push_back(new GL_ParticleFluid);
		samples.push_back(new GL_ViscosityFish);
		//samples.push_back(new QT_GhostSPH);
		samples.push_back(new Qt_ParticleSkinning);

		// Tutorials
		samples.push_back(new GL_GlfwGUI);
		samples.push_back(new GL_PhotorealisticRender);

		// Volume
		samples.push_back(new GL_MarchingCubes);
		samples.push_back(new GL_SDFUniform);
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

WSampleWidget::WSampleWidget(int maxColumns)
{
	//this->setHeight(Wt::WLength("100%"));
	//this->setOverflow(Wt::Overflow::Auto);

	auto container = std::make_unique<Wt::WContainerWidget>();
	auto gridLayout = std::make_unique<Wt::WGridLayout>();

	int row = 0;
	int col = 0;


	for (auto sample : SampleStore::getInstance()->getSamples())
	{
		auto item = _createItem(sample);
		gridLayout->addWidget(std::unique_ptr<Wt::WContainerWidget>(item), row, col);

		item->clicked().connect([=]()
			{
				m_signal.emit(sample);
			});

		col++;
		if (col == maxColumns)
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