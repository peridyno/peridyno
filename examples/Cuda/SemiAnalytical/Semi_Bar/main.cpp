#include "UbiApp.h"

#include "SceneGraph.h"
#include "Log.h"

#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/Emitters/SquareEmitter.h"
#include "ParticleSystem/Emitters/CircularEmitter.h"
#include "ParticleSystem/ParticleFluid.h"

#include "Topology/TriangleSet.h"
#include "Collision/NeighborPointQuery.h"

#include "Module/CalculateNorm.h"

#include <ColorMapping.h>

#include <GLPointVisualModule.h>
#include <GLSurfaceVisualModule.h>
#include <GLInstanceVisualModule.h>

#include <GLRenderEngine.h>

#include "SemiAnalyticalScheme/ComputeParticleAnisotropy.h"
#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"

#include "ParticleSystem/MakeParticleSystem.h"

#include "StaticTriangularMesh.h"

using namespace std;
using namespace dyno;

template<typename TDataType>
class LoadParticles : public ParticleSystem<TDataType>
{
	DECLARE_TCLASS(LoadParticles, TDataType);
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	LoadParticles() {};
	~LoadParticles() override {};

	void loadParticles(Coord center, Real r,
		Coord center1, Real r1,
		Coord center2, Real r2,
		Coord center3, Real r3,
		Coord center4, Real r4,
		Coord center5, Real r5,
		Coord center6, Real r6,
		Coord center7, Real r7,
		Coord center8, Real r8,
		Coord center9, Real r9,
		Coord center10, Real r10,
		Coord center11, Real r11,
		Coord center12, Real r12,
		Coord center13, Real r13,
		Coord center14, Real r14, Real distance)
	{
		std::vector<Coord> vertList;

		Coord lo = center - r;
		Coord hi = center + r;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p = Coord(x, y, z);
					if ((p - center).norm() < r)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center1 - r1;
		hi = center1 + r1;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p1 = Coord(x, y, z);
					if ((p1 - center1).norm() < r1)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center2 - r2;
		hi = center2 + r2;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p2 = Coord(x, y, z);
					if ((p2 - center2).norm() < r2)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center3 - r3;
		hi = center3 + r3;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p3 = Coord(x, y, z);
					if ((p3 - center3).norm() < r3)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center4 - r4;
		hi = center4 + r4;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p4 = Coord(x, y, z);
					if ((p4 - center4).norm() < r4)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center5 - r5;
		hi = center5 + r5;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p5 = Coord(x, y, z);
					if ((p5 - center5).norm() < r5)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center6 - r6;
		hi = center6 + r6;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p6 = Coord(x, y, z);
					if ((p6 - center6).norm() < r6)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center7 - r7;
		hi = center7 + r7;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p7 = Coord(x, y, z);
					if ((p7 - center7).norm() < r7)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center8 - r8;
		hi = center8 + r8;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p8 = Coord(x, y, z);
					if ((p8 - center8).norm() < r8)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center9 - r9;
		hi = center9 + r9;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p9 = Coord(x, y, z);
					if ((p9 - center9).norm() < r9)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center10 - r10;
		hi = center10 + r10;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p10 = Coord(x, y, z);
					if ((p10 - center10).norm() < r10)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center11 - r11;
		hi = center11 + r11;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p11 = Coord(x, y, z);
					if ((p11 - center11).norm() < r11)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center12 - r12;
		hi = center12 + r12;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p12 = Coord(x, y, z);
					if ((p12 - center12).norm() < r12)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center13 - r13;
		hi = center13 + r13;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p13 = Coord(x, y, z);
					if ((p13 - center13).norm() < r13)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		lo = center14 - r14;
		hi = center14 + r14;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p14 = Coord(x, y, z);
					if ((p14 - center14).norm() < r14)
					{
						vertList.push_back(Coord(x, y, z));
					}
				}
			}
		}

		this->statePosition()->assign(vertList);
		this->stateVelocity()->resize(vertList.size());
		this->stateVelocity()->reset();

		vertList.clear();
	};
	
protected:
	void resetStates() override {
		ParticleSystem<TDataType>::resetStates();
	};
};

IMPLEMENT_TCLASS(LoadParticles, TDataType);

std::shared_ptr<SceneGraph> createScene()
{
	//**********Scene
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	//scene.setGravity(Vec3f(0, -7.0, 0));
	scn->setUpperBound(Vec3f(2.5, 2.5, 2.5));
	scn->setLowerBound(Vec3f(-2.5, -2.5, -2.5));

	auto loader = scn->addNode(std::make_shared<LoadParticles<DataType3f>>());
	loader->loadParticles(
		Vec3f(-0.16, 0.15, 0.015), 0.018,
		Vec3f(-0.14, 0.12, 0.06), 0.017,
		Vec3f(-0.13, 0.15, -0.03), 0.014,
		Vec3f(-0.09, 0.15, 0.02), 0.02,
		Vec3f(-0.01, 0.07, 0.075), 0.014,
		Vec3f(0.0, 0.15, 0.02), 0.024,
		Vec3f(0.03, 0.1, 0.07), 0.017,
		Vec3f(0.07, 0.15, 0.05), 0.015,
		Vec3f(0.03, 0.15, -0.045), 0.015,
		Vec3f(0.06, 0.1, -0.07), 0.013,
		Vec3f(-0.08, 0.15, -0.05), 0.019,
		Vec3f(-0.1, 0.2, 0.06), 0.014,
		Vec3f(-0.05, 0.28, -0.03), 0.012,
		Vec3f(-0.02, 0.25, -0.04), 0.015,
		Vec3f(-0.04, 0.24, 0.065), 0.016,
		0.005);

	//Create a particle emitter
	auto emitter = scn->addNode(std::make_shared<SquareEmitter<DataType3f>>());
	emitter->varLocation()->setValue(Vec3f(0.0f, 0.5f, 0.5f));

	//**********Boundary 2
	auto pipe = scn->addNode(std::make_shared<StaticTriangularMesh<DataType3f>>());
	pipe->varFileName()->setValue(getAssetPath() + "bar/pipe.obj");
	pipe->varLocation()->setValue(Vec3f(Vec3f(0.0, 0.0, 0.0)));
	pipe->varScale()->setValue(Vec3f(1 / 5.0));

	auto meshRenderer = std::make_shared<GLSurfaceVisualModule>();
	meshRenderer->setColor(Color(0.26f, 0.25f, 0.25f));
	meshRenderer->setVisible(true);
	pipe->stateTriangleSet()->connect(meshRenderer->inTriangleSet());
	pipe->graphicsPipeline()->pushModule(meshRenderer);

	//**********Solid Fluid Interaction Node
	auto sfi = scn->addNode(std::make_shared<SemiAnalyticalSFINode<DataType3f>>());

	loader->connect(sfi->importInitialStates());
	pipe->stateTriangleSet()->connect(sfi->inTriangleSet());

	//Visualize fluid particles in SemiAnalyticalSFINode
	{
		////neighbor query
		auto nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
		nbrQuery->inRadius()->setValue(0.01);
		sfi->statePosition()->connect(nbrQuery->inPosition());
		sfi->graphicsPipeline()->pushModule(nbrQuery);

		//**********particle rendering with anisotropic values
		//anisotropic value calculator
		auto m_eigenV = std::make_shared<ComputeParticleAnisotropy<DataType3f>>();
		m_eigenV->varSmoothingLength()->setValue(0.01);
		sfi->statePosition()->connect(m_eigenV->inPosition());
		nbrQuery->outNeighborIds()->connect(m_eigenV->inNeighborIds());
		sfi->graphicsPipeline()->pushModule(m_eigenV);

		auto instanceRender = std::make_shared<GLInstanceVisualModule>();
		m_eigenV->outTransform()->connect(instanceRender->inInstanceTransform());
		//instanceRender->inTriangleSet()->allocate();
		std::shared_ptr<TriangleSet<DataType3f>> triSet = std::make_shared<TriangleSet<DataType3f>>();
		triSet->loadObjFile(getAssetPath() + "standard/standard_icosahedron.obj");
		instanceRender->inTriangleSet()->setDataPtr(triSet);

		sfi->graphicsPipeline()->pushModule(instanceRender);
	}

	return scn;
}

int main()
{
	UbiApp window(GUIType::GUI_QT);
	window.setSceneGraph(createScene());
	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}


