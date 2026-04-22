#include <UbiApp.h>
#include <SceneGraph.h>

#include <Volume/Module/VolumeToTriangleSet.h>

#include <Module/ComputeModule.h>

#include <EulerFluid/EulerianFluid3D.h>

#include <Topology/LevelSet.h>

#include <GLSurfaceVisualModule.h>

using namespace std;
using namespace dyno;

template<typename TDataType>
class PhaseFieldToVolume : public ComputeModule
{
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
public:
	PhaseFieldToVolume() {};
	~PhaseFieldToVolume() override {};

	DEF_INSTANCE_IN(PhaseField<TDataType>, PhaseField, "");
	DEF_INSTANCE_OUT(LevelSet<TDataType>, LevelSet, "");
private:

	void compute() override {
		if (this->outLevelSet()->isEmpty())
		{
			this->outLevelSet()->allocate();
		}

		auto phasefield = this->inPhaseField()->constDataPtr();
		auto levelset = this->outLevelSet()->getDataPtr();

		auto& sdf = levelset->getSDF();

		auto p0 = phasefield->origin();
		auto p1 = p0 + Coord(0.01 * phasefield->nx(), 0.01 * phasefield->ny(), 0.01 * phasefield->nz());
		sdf.setSpace(p0, p1, 0.01);

		sdf.distances().assign(phasefield->volumeFraction());
	}
};


std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setLowerBound(Vec3f(-1.0f, 0.0f, -0.1f));
	scn->setUpperBound(Vec3f(1.0f, 2.0f, 0.1f));

	//create a node representing 3d Eulerian fluids.
	auto fluid3d = scn->addNode(std::make_shared<EulerianFluid3D<DataType3f>>());
	auto p2v = std::make_shared<PhaseFieldToVolume<DataType3f>>();
	fluid3d->statePhaseField()->connect(p2v->inPhaseField());
	fluid3d->graphicsPipeline()->pushModule(p2v);

	auto v2t = std::make_shared<VolumeToTriangleSet<DataType3f>>();
	v2t->varIsoValue()->setValue(0.5f);
	p2v->outLevelSet()->connect(v2t->inVolume());
	fluid3d->graphicsPipeline()->pushModule(v2t);

	auto render = std::make_shared<GLSurfaceVisualModule>();
	v2t->outTriangleSet()->connect(render->inTriangleSet());
	fluid3d->graphicsPipeline()->pushModule(render);

	return scn;
}

int main()
{
	UbiApp window(GUIType::GUI_QT);

	window.setSceneGraph(createScene());
	window.initialize(1280, 768);

	window.mainLoop();

	return 0;
}


