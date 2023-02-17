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
 * @brief This example demonstrates the connection between nodes.
 */

class Data : public Object {
public:
	Data() {};
	~Data() override {};
};

class Source : public Node
{
	DECLARE_CLASS(Source);
public:
	Source() {
	};
	~Source() override {};

	DEF_INSTANCE_OUT(Data, Data, "Output");

protected:
	void resetStates() {};
};

IMPLEMENT_CLASS(Source);

class Target : public Node
{
	DECLARE_CLASS(Target);
public:
	Target() {
	};
	~Target() override {};

	DEF_NODE_PORTS(Source, Source, "");

	DEF_INSTANCE_IN(Data, Data, "Input");

protected:
	void resetStates() {};
};

IMPLEMENT_CLASS(Target);

int main()
{
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	auto source1 = scn->addNode(std::make_shared<Source>());
	source1->setName("Source1");

	auto source2 = scn->addNode(std::make_shared<Source>());
	source2->setName("Source2");

	auto target1 = scn->addNode(std::make_shared<Target>());
	target1->setName("Target");

	source1->outData()->connect(target1->inData());

	QtApp app;
	app.setSceneGraph(scn);
	app.initialize(1366, 800);
	app.mainLoop();

	return 0;
}