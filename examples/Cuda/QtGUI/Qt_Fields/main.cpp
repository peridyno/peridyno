/**
 * Copyright 2023 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <QtApp.h>
using namespace dyno;

#include "Node.h"
#include "Vector.h"

/**
 * @brief This example demonstrates how to define fields for a node.
 */

class Fields : public Node
{
	DECLARE_CLASS(Fields);
public:
	Fields() {
		this->varFloat()->setRange(0.01, 10.0f);
		this->inFloatArray()->tagOptional(true);
	};
	~Fields() {};

public:
	DEF_VAR(bool, Boolean, false, "Define a boolean field");

	DEF_VAR(int, Int, 1, "Define an int");

	DEF_VAR(float, Float, 1.0f, "Define a float field");

	DEF_VAR(Vec3f, Vector, Vec3f(1.0f), "Define a vector field");

	DEF_ARRAY_IN(float, FloatArray, DeviceType::GPU, "Define a float array as input");

	DEF_ARRAY_OUT(float, FloatArray, DeviceType::GPU, "Define a float array as output");

	DEF_ARRAY_STATE(float, Value, DeviceType::GPU, "Define a float array as state");

protected:
	void resetStates() {
		std::cout << "resetStates() " << " is called " << std::endl;
	}
};

IMPLEMENT_CLASS(Fields);

int main()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	auto source1 = scn->addNode(std::make_shared<Fields>());

	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1366, 800);
	app.mainLoop();

	return 0;
}