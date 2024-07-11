#include <QtApp.h>
#include <GLRenderEngine.h>

#include "Auxiliary/DataSource.h"
#include "Auxiliary/DebugInfo.h"
#include "Topology/TriangleSet.h"

using namespace dyno;

/**
 * @brief This example demonstrates how to use data sources inside PeriDyno.
 */

class Source : public Node
{
	DECLARE_CLASS(Source);
public:


	void varChanged() 
	{
		auto v = this->varTestVectorInt()->getValue();
		printf("field Value: ");
		for (size_t i = 0; i < v.size(); i++)
		{
			std::cout << v[i] << ", ";
		}
		std::cout << std::endl;
	}

	DEF_VAR(std::vector<int>, TestVectorInt, std::vector<int>(3,0), "");

	DEF_ARRAY_STATE(Vec3f, Float, DeviceType::GPU, "");
	DEF_ARRAY_STATE(Vec3d, Double, DeviceType::GPU, "");
	DEF_ARRAY_STATE(int, Int, DeviceType::GPU, "");
	DEF_ARRAY_STATE(uint, Uint, DeviceType::GPU, "");
	DEF_ARRAY_STATE(Vec3f, Vec2Float, DeviceType::GPU, "");

	DEF_ARRAYLIST_STATE(Transform3f, InstanceTransform, DeviceType::GPU, "Instance transforms");
	DEF_ARRAYLIST_STATE(Transform3f, InstanceTransformCPU, DeviceType::CPU, "Instance transforms");
	DEF_ARRAY_STATE(Transform3f, ArrayTransform, DeviceType::GPU, "Instance transforms");
	DEF_VAR_STATE(Transform3f, VarTransform,Transform3f(Vec3f(1.25),Mat3f::identityMatrix(),Vec3f(1.0)), "Instance transforms");

	DEF_VAR_STATE(Vec3d, VarDouble, Vec3d(0.9, 1.0, 0.8), "");
	DEF_INSTANCE_STATE(TriangleSet<DataType3f>, TriangleSet, "");
	DEF_INSTANCE_STATE(PointSet<DataType3f>, PointSet, "");
	DEF_INSTANCE_STATE(EdgeSet<DataType3f>, EdgeSet, "");
	DEF_INSTANCE_STATE(PointSet<DataType3f>, PointSetS, "");

	void updateStates() override
	{
		{
			auto df = this->stateFloat()->getDataPtr();
			CArray<Vec3f> cf;
			cf.assign(*df);

			for (size_t i = 0; i < cf.size(); i++)
			{
				for (size_t j = 0; j < 3; j++)
				{
					cf[i][j] ++;

				}
			}
			df->assign(cf);
		}

		{
			auto dd = this->stateDouble()->getDataPtr();
			CArray<Vec3d> cd;
			cd.assign(*dd);

			for (size_t i = 0; i < cd.size(); i++)
			{
				for (size_t j = 0; j < 3; j++)
				{
					cd[i][j] ++;
				}
			}
			dd->assign(cd);
		}

		{
			auto dint = this->stateInt()->getDataPtr();
			CArray<int> cint;
			cint.assign(*dint);

			for (size_t i = 0; i < cint.size(); i++)
			{
				cint[i] ++;
			}
			dint->assign(cint);
		}


		auto ptSet = this->statePointSet()->getDataPtr();
		CArray<Vec3f> cpts;
		cpts.assign(ptSet->getPoints());

		for (size_t i = 0; i < cpts.size(); i++)
		{
			cpts[i][0] = cpts[i][0] + 1;
		}
		
		DArray<Vec3f> dpts;
		dpts.assign(cpts);

		ptSet->setPoints(dpts);


	}

	void resetStates()override
	{
		initialData();
		varChanged();
	}

	void initialData()
	{
		std::vector<Vec3f> v3f = 
		{ 
			Vec3f(1.0,1.0,1.3),Vec3f(2.22,3.33,4.44),
			Vec3f(1.0,1.0,1.3),Vec3f(2.22,3.33,4.44),
			Vec3f(1.0,1.0,1.3),Vec3f(2.22,3.33,4.44),
			Vec3f(1.0,1.0,1.3),Vec3f(2.22,3.33,4.44),
			Vec3f(1.0,1.0,1.3),Vec3f(2.22,3.33,4.44),
			Vec3f(1.0,1.0,1.3),Vec3f(2.22,3.33,4.44),
			Vec3f(1.0,1.0,1.3),Vec3f(2.22,3.33,4.44),
			Vec3f(1.0,1.0,1.3),Vec3f(2.22,3.33,4.44),
			Vec3f(1.0,1.0,1.3),Vec3f(2.22,3.33,4.44),
			Vec3f(1.0,1.0,1.3),Vec3f(2.22,3.33,4.44)
			
		};
		this->stateFloat()->assign(v3f);

		std::vector<Vec3d> v3d;

		for (size_t i = 0; i < 100000; i++)
		{
			v3d.push_back(Vec3d(i,0,0));
		}

		this->stateDouble()->assign(v3d);

		std::vector<int> intfield = { -3,-2,-1,0,1,2,3,4,5 };
		this->stateInt()->assign(intfield);

		std::vector<uint> uintfield ;
		for (size_t i = 0; i < 10000; i++)
		{
			uintfield.push_back(i);
		}
		this->stateUint()->assign(uintfield);


		// Transform3f
		// ArrayList
		CArrayList<Transform3f> tms;
		CArray<uint> counter(3);
		counter[0] = 1;
		counter[1] = 3;
		counter[2] = 7;

		tms.resize(counter);
		uint index = 0;
		for (uint i = 0; i < counter.size(); i++)
		{
			auto it = counter[i];
			auto& list = tms[index];
			for (size_t j = 0; j < it; j++)
			{			
				list.insert(Transform3f(Vec3f(float(it) + 0.12345), Mat3f::identityMatrix(), Vec3f(it)));
			}
			index++;
		}
		index = 0;

		if (this->stateInstanceTransform()->isEmpty())
		{
			this->stateInstanceTransform()->allocate();
		}

		auto instantanceTransform = this->stateInstanceTransform()->getDataPtr();
		instantanceTransform->assign(tms);

		if (this->stateInstanceTransformCPU()->isEmpty())
		{
			this->stateInstanceTransformCPU()->allocate();
		}
		auto instantanceTransformCPU = this->stateInstanceTransformCPU()->getDataPtr();
		instantanceTransformCPU->assign(tms);


		tms.clear();



		// Array
		CArray<Transform3f> c_Trans;
		for (size_t i = 0; i < 15; i++)
			c_Trans.pushBack(Transform3f(Vec3f(i,1,1), Mat3f::identityMatrix(), Vec3f(0.5,i,1)));

		this->stateArrayTransform()->assign(c_Trans);


		std::vector<Vec3f> pt = { Vec3f(0,0,0),Vec3f(0,1,0), Vec3f(1,1,0),Vec3f(1,0,0) };
		this->statePointSet()->getDataPtr()->setPoints(pt);
		this->statePointSetS()->getDataPtr()->setPoints(pt);;
	}


	Source() {
		auto floating = std::make_shared<FloatingNumber<DataType3f>>();

		auto printFloat = std::make_shared<PrintFloat>();

		floating->outFloating()->connect(printFloat->inFloat());

		this->animationPipeline()->pushModule(floating);
		this->animationPipeline()->pushModule(printFloat);

		this->statePointSet()->setDataPtr(std::make_shared<PointSet<DataType3f>>());
		this->statePointSetS()->setDataPtr(std::make_shared<PointSet<DataType3f>>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&Source::varChanged, this));

		this->varTestVectorInt()->attach(callback);

	};

	~Source() override {};
};

IMPLEMENT_CLASS(Source);


int main(int, char**)
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	//Create a sphere
	auto src = scn->addNode(std::make_shared<Source>());


	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1024, 768);
	app.mainLoop();

	return 0;
}
