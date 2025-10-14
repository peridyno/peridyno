#include <UbiApp.h>

#include <SceneGraph.h>

#include <HeightField/LandScape.h>
#include <HeightField/MountainTorrents.h>
#include <HeightField/Module/NumericalScheme.h>

#include <Mapping/HeightFieldToTriangleSet.h>

#include <GLSurfaceVisualModule.h>


#include <vector>
#include <cmath>
#include <random>


using namespace std;
using namespace dyno;

void generateUpstreamMountainTerrain(CArray2D<float>& heights, int start, int end, int ny) {
	float maxHeight = 2.0;
	for (int i = start; i < end; ++i) {
		float factor = static_cast<float>(end - i) / end;
		for (int j = 0; j < ny; ++j) {
			heights(i, j) = maxHeight * factor;
		}
	}
}

void generateUpstreamCanyonTerrain(CArray2D<float>& heights, int start, int end, int nx, int ny) {

	float maxHeight = 2.0;
	int canyonWidth = 80;

	int canyonLeft = (ny - canyonWidth) / 2;
	int canyonRight = canyonLeft + canyonWidth;

	for (int i = 0; i < nx; ++i) {
		float xFactor = std::pow(static_cast<float>(nx - i) / nx, 2);

		for (int j = 0; j < ny; ++j) {
			if (j >= canyonLeft && j < canyonRight) {
				heights(i, j) = 0.1 * xFactor;
			}
			else {
				float distanceToCanyonEdge = std::min(std::abs(j - canyonLeft), std::abs(j - canyonRight));
				float mountainFactor = std::exp(-std::pow(distanceToCanyonEdge / 30.0, 2));
				heights(i, j) = maxHeight * mountainFactor * xFactor;
			}
		}
	}
}

void generateDownstreamCityTerrain(CArray2D<float>& heights, int start, int end, int ny) {
	const float amplitude1 = 0.02;
	const float frequency1 = 0.1;
	const float amplitude2 = 0.02;
	const float frequency2 = 0.1;
	for (int i = start; i < end; ++i) {
		for (int j = 0; j < ny; ++j) {
			float height = amplitude1 * std::sin(frequency1 * i) + amplitude2 * std::cos(frequency2 * j);
			heights(i, j) = height;
		}
	}
}

void addDownstreamBuildingsAndTrees(CArray2D<float>& heights, int start, int nx, int ny) {
	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_int_distribution<> buildingXDist(start, nx - 1);
	std::uniform_int_distribution<> buildingYDist(0, ny - 1);
	std::uniform_int_distribution<> buildingSizeDist(5, 15);
	std::uniform_real_distribution<> buildingHeightDist(0.5, 3.0);

	for (int i = 0; i < 20; ++i) {
		int x = buildingXDist(gen);
		int y = buildingYDist(gen);
		int size = buildingSizeDist(gen);
		float height = buildingHeightDist(gen);
		for (int dx = x; dx < std::min(x + size, static_cast<int>(nx)); ++dx) {
			for (int dy = y; dy < std::min(y + size, static_cast<int>(ny)); ++dy) {
				heights(dx, dy) += height;
			}
		}
	}

	std::uniform_int_distribution<> treeXDist(start, nx - 1);
	std::uniform_int_distribution<> treeYDist(0, ny - 1);
	std::uniform_int_distribution<> treeSizeDist(3, 6);
	std::uniform_real_distribution<> treeHeightDist(0.125, 0.5);

	for (int i = 0; i < 30; ++i) {
		int x = treeXDist(gen);
		int y = treeYDist(gen);
		int size = treeSizeDist(gen);
		float height = treeHeightDist(gen);
		float radius = static_cast<float>(size) / 2;
		for (int dx = std::max(0, x - size); dx < std::min(x + size, static_cast<int>(nx)); ++dx) {
			for (int dy = std::max(0, y - size); dy < std::min(y + size, static_cast<int>(ny)); ++dy) {
				float dist = std::sqrt(static_cast<float>((dx - x) * (dx - x) + (dy - y) * (dy - y)));
				if (dist <= radius) {
					float factor = (1 + std::cos((dist / radius) * M_PI)) / 2;
					heights(dx, dy) += height * factor;
				}
			}
		}
	}
}



std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto land = scn->addNode(std::make_shared<LandScape<DataType3f>>());
	land->varFileName()->setValue(getAssetPath() + "landscape/Landscape_1_Map_1024x1024.png");
	land->varLocation()->setValue(Vec3f(0.0f, 100.0f, 0.0f));
	land->varScale()->setValue(Vec3f(1.0f, 64.0f, 1.0f)); 

	return scn;
}

std::shared_ptr<SceneGraph> createSceneWater()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

	auto land = scn->addNode(std::make_shared<LandScape<DataType3f>>());
	land->varLocation()->setValue(Vec3f(0.0f, 100.0f, 0.0f));
	land->varScale()->setValue(Vec3f(1.0f, 64.0f, 1.0f));


	CArray2D<float> heights;
	uint nx = 512, ny = 256;
	heights.resize(nx, ny);

	int mountainEnd = 256;
	generateUpstreamMountainTerrain(heights, 0, mountainEnd, ny);
	generateUpstreamCanyonTerrain(heights, 0, mountainEnd, nx, ny);
	generateDownstreamCityTerrain(heights, mountainEnd, nx, ny);
	addDownstreamBuildingsAndTrees(heights, mountainEnd, nx, ny);

	land->stateInitialHeights()->assign(heights);

	auto water = scn->addNode(std::make_shared<MountainTorrents<DataType3f>>());
	
	CArray2D<float> waterHeights(nx, ny);
	for (int x = 0; x < nx; x++)
	{
		for (int y = 0; y < ny; y++)
		{
			if (x <= 256 && (y >= 88 && y < 168))
				waterHeights(x, y) = 20;
			else
				waterHeights(x, y) = 0;
		}
	}

	water->varWaterLevel()->setValue(0.0f);
	water->varViscosity()->setValue(0.1);
	water->stateInitialHeights()->assign(waterHeights);

	land->connect(water->importTerrain());

	heights.clear();
	waterHeights.clear();

	return scn;
}



int main()
{
    Vec4f glx(5.0f, 5.0f, 0.0f, 0.0f);
	Vec4f grx(5.0f, 5.0f, 0.0f, 0.0f);
	auto flux = FirstOrderUpwindX(glx, grx, 9.8f, 0.001f);
    

	UbiApp app(GUIType::GUI_GLFW);

	app.setSceneGraph(createSceneWater());
	app.initialize(1024, 768);

	app.renderWindow()->getCamera()->setUnitScale(512);

	app.mainLoop();

        
	return 0;
}