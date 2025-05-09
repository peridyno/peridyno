#include "SceneLoaderXML.h"
#include <algorithm>

#include <stdlib.h>
#include <sstream>

namespace dyno
{
	using LoadResult = std::variant<std::shared_ptr<SceneGraph>, std::string>;

	LoadResult SceneLoaderXML::load(const std::string filename)
	{
		tinyxml2::XMLDocument doc;
		if (doc.LoadFile(filename.c_str()))
		{
			doc.PrintError();
			return "Error Load";
		}

		for (size_t i = 0; i < mConnectionInfo.size(); i++) {
			mConnectionInfo[i].clear();
		}
		mConnectionInfo.clear();

		std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();

		tinyxml2::XMLElement* root = doc.RootElement();

<<<<<<< HEAD
		tinyxml2::XMLElement* versionXML = root->FirstChildElement("Version");
		std::string version = TOSTRING(PERIDYNO_VERSION);
		if (version.compare(versionXML->GetText()) != 0)
		{
			return "Error Version";
		}

=======
		scn->setUpperBound(decodeVec3f(root->Attribute("UpperBound")));
		scn->setLowerBound(decodeVec3f(root->Attribute("LowerBound")));
>>>>>>> public

		std::vector<std::shared_ptr<Node>> nodes;
		tinyxml2::XMLElement* nodeXML = root->FirstChildElement("Node");

		std::map<std::shared_ptr<Node>, tinyxml2::XMLElement*> node2xml;

		while (nodeXML)
		{
			auto node = processNode(nodeXML);
			if (node) {
				scn->addNode(node);
				nodes.push_back(node);
			}
			node2xml[node] = nodeXML;

			nodeXML = nodeXML->NextSiblingElement("Node");
		}

		SceneGraph::Iterator itor = scn->begin();

		std::map<ObjectId, std::shared_ptr<Node>> objID2Node;
		std::map<uint, ObjectId> index2ObjectId;
		int radix = 0;

		for (auto it : nodes) {

			objID2Node[it->objectId()] = it;
			index2ObjectId[radix] = it->objectId();
			radix++;
		}

		for (auto it : node2xml)
		{
			ConstructNodePipeline(it.first,it.second, nodes);
		}

		for (size_t i = 0; i < mConnectionInfo.size(); i++)
		{
			auto& connections = mConnectionInfo[i];
			for (size_t j = 0; j < connections.size(); j++)
			{
				auto info = connections[j];

				auto expNode = nodes[info.src];
				auto inpNode = nodes[info.dst];

				auto& outFields = expNode->getOutputFields();
				auto& FieldsSrc = expNode->getAllFields();

				auto& inPorts = inpNode->getImportNodes();
				auto& inFields = inpNode->getInputFields();

				if (info.srcModuleId != -1) //from module;
				{
					std::shared_ptr<Pipeline> pipline;
					switch (info.srcModulePipline)
					{
					case 0:
						pipline = expNode->animationPipeline();
						break;
					case 1:
						pipline = expNode->graphicsPipeline();
						break;
					case 2:
						pipline = expNode->resetPipeline();
						break;
					default:
						break;
					}
					auto& activemodules = pipline->activeModules();
					std::vector<std::shared_ptr<Module>> modules;
					for (const auto& elem : activemodules) {
						modules.push_back(elem);
					}
					FBase* fout = modules[info.srcModuleId]->getAllFields()[info.id0];
					FBase* fin = inpNode->getAllFields()[info.id1];

					fout->connect(fin);
					if (fin->getFieldType() != FieldTypeEnum::Out)
						pipline->promoteOutputToNode(fout);
				}
				else
				{
					if (info.id0 <= 0)
					{
						expNode->connect(inPorts[info.id1]);
					}
					else
					{
						FieldsSrc[info.id0]->connect(inpNode->getAllFields()[info.id1]);
					}
				}

			}
		}

		return scn;
	}

	bool SceneLoaderXML::save(std::shared_ptr<SceneGraph> scn, const std::string filename)
	{
		tinyxml2::XMLDocument doc;

		// 		const char* declaration = "<?xml version=\"0.6.0\" encoding=\"UTF-8\">";
		// 		doc.Parse(declaration);
<<<<<<< HEAD

=======
>>>>>>> public

		tinyxml2::XMLElement* root = doc.NewElement("SceneGraph");

		root->SetAttribute("LowerBound", encodeVec3f(scn->getLowerBound()).c_str());
		root->SetAttribute("UpperBound", encodeVec3f(scn->getUpperBound()).c_str());

		doc.InsertEndChild(root);

		// version
		tinyxml2::XMLElement* version = doc.NewElement("Version");
		version->SetText(TOSTRING(PERIDYNO_VERSION));
		root->InsertEndChild(version);


		SceneGraph::Iterator itor = scn->begin();

		std::map<ObjectId, uint> indices;
		int radix = 0;
		for (; itor != scn->end(); itor++) {
			auto node = itor.get();
			indices[node->objectId()] = radix;

			radix++;
		}
		for (itor = scn->begin(); itor != scn->end(); itor++)
		{
			auto node = itor.get();

			tinyxml2::XMLElement* nodeXml = doc.NewElement("Node");
			nodeXml->SetAttribute("Class", node->getClassInfo()->getClassName().c_str());
			nodeXml->SetAttribute("Coordinate", encodeVec2f(Vec2f(node->bx(), node->by())).c_str());
			root->InsertEndChild(nodeXml);

			tinyxml2::XMLElement* varsXml = doc.NewElement("Variables");
			nodeXml->InsertEndChild(varsXml);

			//write control variables
			auto& params = node->getParameters();
			for (auto var : params)
			{
				tinyxml2::XMLElement* field = doc.NewElement("Field");
				field->SetAttribute("Name", var->getObjectName().c_str());

				tinyxml2::XMLText* val = doc.NewText(var->serialize().c_str());
				field->InsertEndChild(val);

				varsXml->InsertEndChild(field);
			}

			//write connections
			tinyxml2::XMLElement* nodeConnectionsXml = doc.NewElement("Connections");
			nodeXml->InsertEndChild(nodeConnectionsXml);

			auto ports = node->getImportNodes();

			for (int i = 0; i < ports.size(); i++)
			{
				auto NodesSrc = ports[i]->getNodes();
				bool fieldFound = false;
				Node* nSrcPar = nullptr;
				for (auto nSrc : NodesSrc)
				{
					auto nSrcExports = nSrc->getExportNodes();
					for (auto x : nSrcExports)
					{
						if (x == ports[i])
						{
							fieldFound = true;
							nSrcPar = nSrc;

							tinyxml2::XMLElement* connection = doc.NewElement("Connection");
							connection->SetAttribute("SourceId", indices[nSrcPar->objectId()]);
							connection->SetAttribute("From", 0);
							connection->SetAttribute("TargetId", indices[node->objectId()]);
							connection->SetAttribute("To", uint(i));
							nodeConnectionsXml->InsertEndChild(connection);
						}
					}
				}
<<<<<<< HEAD
				if (fieldFound && nSrcPar != nullptr) {

					tinyxml2::XMLElement* connection = doc.NewElement("Connection");
					connection->SetAttribute("SourceId", indices[nSrcPar->objectId()]);
					connection->SetAttribute("From", 0);
					connection->SetAttribute("TargetId", indices[node->objectId()]);
					connection->SetAttribute("To", uint(i));
					nodeConnectionsXml->InsertEndChild(connection);
				}

=======
>>>>>>> public
			}

			auto fieldInp = node->getInputFields();

			for (int i = 0; i < fieldInp.size(); i++)
			{
				uint ToIndex = 0;
				auto InNodeFields = node->getAllFields();
				for (ToIndex = 0; ToIndex < InNodeFields.size(); ToIndex++)
				{
					if (fieldInp[i] == InNodeFields[ToIndex])
						break;
				}
				auto fieldSrc = fieldInp[i]->getSource();
				if (fieldSrc != nullptr) {
					auto parSrc = fieldSrc->parent();
					if (parSrc != nullptr)
					{
<<<<<<< HEAD
						Node* nodeSrc = dynamic_cast<Node*>(parSrc);
						auto outId = nodeSrc->objectId();
						auto fieldsOut = nodeSrc->getOutputFields();
						uint outFieldIndex = 0;
						bool fieldFound = false;

						for (outFieldIndex = 0; outFieldIndex < fieldsOut.size(); outFieldIndex++)
=======
						Node* nodeSrc = dynamic_cast<Node*>(parSrc);// [node->field] - [node->field]
						if (nodeSrc != nullptr)
>>>>>>> public
						{
							auto fieldsSrc = nodeSrc->getAllFields();

							bool fieldFound = false;

							uint FromIndex = 0;
							for (FromIndex = 0; FromIndex < fieldsSrc.size(); FromIndex++)
							{
								if (fieldsSrc[FromIndex] == fieldSrc)
								{
									fieldFound = true;
									break;
								}
							}


							if (fieldFound) {
								tinyxml2::XMLElement* connection = doc.NewElement("Connection");
								connection->SetAttribute("SourceId", indices[parSrc->objectId()]);

								connection->SetAttribute("TargetId", indices[node->objectId()]);
								connection->SetAttribute("To", ToIndex);
								connection->SetAttribute("From", FromIndex);
								nodeConnectionsXml->InsertEndChild(connection);
							}
						}
<<<<<<< HEAD
						if (fieldFound) {
							tinyxml2::XMLElement* connection = doc.NewElement("Connection");
							connection->SetAttribute("SourceId", indices[parSrc->objectId()]);
							connection->SetAttribute("From", 1 + outFieldIndex);
							connection->SetAttribute("TargetId", indices[node->objectId()]);
							connection->SetAttribute("To", uint(i + ports.size()));
							nodeConnectionsXml->InsertEndChild(connection);
=======


						Module* moduleSrc = dynamic_cast<Module*>(parSrc);// [node->Module->field] - [node->field]
						if (moduleSrc != nullptr)
						{
							std::map<ObjectId, uint> ModuleIndices;
							auto getModuleData = [&](Module* module, int& index, int& channel)
							{
								{
									uint radix = 0;

									auto& animModules = module->getParentNode()->animationPipeline()->activeModules();
									for (auto m : animModules) {
										if (module == m.get())
										{
											channel = 0;
											index = radix;
											return;
										}
										radix++;
									}

									radix = 0;
									auto& renderModules = module->getParentNode()->graphicsPipeline()->activeModules();
									for (auto m : renderModules) {
										if (module == m.get())
										{
											channel = 1;
											index = radix;
											return;
										}
										radix++;
									}

									radix = 0;
									auto& resetModules = module->getParentNode()->resetPipeline()->activeModules();
									for (auto m : resetModules) {
										if (module == m.get())
										{
											channel = 2;
											index = radix;
											return;
										}
										radix++;
									}
								}
							};


							auto outId = moduleSrc->objectId();
							auto moduleFieldsSrc = moduleSrc->getAllFields();

							uint Index = 0;
							bool fieldFound = false;
							for (auto f : moduleFieldsSrc)
							{
								if (f == fieldSrc)
								{
									fieldFound = true;
									break;
								}
								Index++;
							}



							if (fieldFound) {
								tinyxml2::XMLElement* connection = doc.NewElement("Connection");

								connection->SetAttribute("SourceId", indices[moduleSrc->getParentNode()->objectId()]);

								int moduleIndex = -1;
								int pipline = -1;
								getModuleData(moduleSrc, moduleIndex, pipline);
								connection->SetAttribute("SourceModuleId", moduleIndex);
								connection->SetAttribute("SourceModulePipline", pipline);
								connection->SetAttribute("TargetId", indices[node->objectId()]);
								connection->SetAttribute("To", ToIndex);
								connection->SetAttribute("From", Index);
								nodeConnectionsXml->InsertEndChild(connection);
							}
>>>>>>> public
						}
					}
				}
			}
<<<<<<< HEAD

=======
>>>>>>> public

			for (auto s : node->getOutputFields())
			{
				auto fieldSrc = s->getSource();
				if (fieldSrc != nullptr)
				{
<<<<<<< HEAD
					fieldsOut.push_back(field);
				}
				else if (field->getFieldType() == dyno::FieldTypeEnum::In)
				{
					auto fieldSrc = field->getSource();
					if (fieldSrc != nullptr) {

						auto parSrc = fieldSrc->parent();
						if (parSrc != nullptr)
=======
					auto parSrc = fieldSrc->parent();
					int ToIndex = -1;
					auto fields = node->getAllFields();
					for (size_t i = 0; i < fields.size(); i++)
					{
						auto t = fields[i];
						if (t == s)
>>>>>>> public
						{
							ToIndex = i;
							break;
						}
					}
					Node* nodeSrc = dynamic_cast<Node*>(parSrc);// [node->field] - [node->field]
					if (nodeSrc != nullptr)
					{
						auto fieldsSrc = nodeSrc->getAllFields();

						bool fieldFound = false;

						uint FromIndex = 0;
						for (FromIndex = 0; FromIndex < fieldsSrc.size(); FromIndex++)
						{
							if (fieldsSrc[FromIndex] == fieldSrc)
							{
								fieldFound = true;
								break;
							}
						}


						if (fieldFound) {
							tinyxml2::XMLElement* connection = doc.NewElement("Connection");
							connection->SetAttribute("SourceId", indices[parSrc->objectId()]);

							connection->SetAttribute("TargetId", indices[node->objectId()]);
							connection->SetAttribute("To", ToIndex);
							connection->SetAttribute("From", FromIndex);
							nodeConnectionsXml->InsertEndChild(connection);
						}
					}


					Module* moduleSrc = dynamic_cast<Module*>(parSrc);// [node->Module->field] - [node->field]
					if (moduleSrc != nullptr)
					{
						std::map<ObjectId, uint> ModuleIndices;
						auto getModuleData = [&](Module* module, int& index, int& channel)
						{

							{
								uint radix = 0;

								auto& animModules = module->getParentNode()->animationPipeline()->activeModules();
								for (auto m : animModules) {
									if (module == m.get())
									{
										channel = 0;
										index = radix;
										return;
									}
									radix++;
								}

								radix = 0;
								auto& renderModules = module->getParentNode()->graphicsPipeline()->activeModules();
								for (auto m : renderModules) {
									if (module == m.get())
									{
										channel = 1;
										index = radix;
										return;
									}
									radix++;
								}

								radix = 0;
								auto& resetModules = module->getParentNode()->resetPipeline()->activeModules();
								for (auto m : resetModules) {
									if (module == m.get())
									{
										channel = 2;
										index = radix;
										return;
									}
									radix++;
								}
							}
						};


						auto outId = moduleSrc->objectId();
						auto moduleFieldsSrc = moduleSrc->getAllFields();

						uint Index = 0;
						bool fieldFound = false;
						for (auto f : moduleFieldsSrc)
						{
							if (f == fieldSrc)
							{
								fieldFound = true;
								break;
							}
							Index++;
						}



						if (fieldFound) {
							tinyxml2::XMLElement* connection = doc.NewElement("Connection");

							connection->SetAttribute("SourceId", indices[moduleSrc->getParentNode()->objectId()]);

							int moduleIndex = -1;
							int pipline = -1;
							getModuleData(moduleSrc, moduleIndex, pipline);
							connection->SetAttribute("SourceModuleId", moduleIndex);
							connection->SetAttribute("SourceModulePipline", pipline);
							connection->SetAttribute("TargetId", indices[node->objectId()]);
							connection->SetAttribute("To", ToIndex);
							connection->SetAttribute("From", Index);
							nodeConnectionsXml->InsertEndChild(connection);
						}
					}

				}
			}
			/**/
			//Insert animation pipeline
			auto savePipeline = [&](std::shared_ptr<Pipeline> pipeline, const char* tag) {
				tinyxml2::XMLElement* pipelineXml = doc.NewElement(tag);
				nodeXml->InsertEndChild(pipelineXml);

				std::map<ObjectId, uint> ModuleIndices;
				std::map<ObjectId, std::shared_ptr<Module>> moduleMap;
				uint radix = 0;

				auto& activeModules = pipeline->activeModules();
				for (auto m : activeModules) {
					tinyxml2::XMLElement* moduleXml = doc.NewElement("Module");
					moduleXml->SetAttribute("Class", m->getClassInfo()->getClassName().c_str());
					moduleXml->SetAttribute("Coordinate", encodeVec2f(Vec2f(m->bx(), m->by())).c_str());
					pipelineXml->InsertEndChild(moduleXml);

					tinyxml2::XMLElement* varsModuleXml = doc.NewElement("Variables");
					moduleXml->InsertEndChild(varsModuleXml);

					//write control variables
					auto& params = m->getParameters();
					for (auto var : params)
					{
						tinyxml2::XMLElement* field = doc.NewElement("Field");
						field->SetAttribute("Name", var->getObjectName().c_str());

						tinyxml2::XMLText* val = doc.NewText(var->serialize().c_str());
						field->InsertEndChild(val);

						varsModuleXml->InsertEndChild(field);
					}

					ModuleIndices[m->objectId()] = radix;
					moduleMap[m->objectId()] = m;
					radix++;
				}
				//write connections
				tinyxml2::XMLElement* moduleConnectionsXml = doc.NewElement("Connections");
				pipelineXml->InsertEndChild(moduleConnectionsXml);

				for (auto m : activeModules)
				{
					auto& fieldIn = m->getInputFields();
					for (int i = 0; i < fieldIn.size(); i++)
					{
						auto fieldSrc = fieldIn[i]->getSource();
						if (fieldSrc != nullptr) {
							auto parSrc = fieldSrc->parent();
							if (parSrc != nullptr)
							{
								uint inFieldIndex = 0;
								bool infieldFound = false;

								auto infields = m->getAllFields();
								for (auto f : infields)
								{
									if (f == fieldIn[i])
									{
										infieldFound = true;
										break;
									}
									inFieldIndex++;
								}

								Module* src = dynamic_cast<Module*>(parSrc);
								if (src != nullptr)
								{
									auto outId = src->objectId();
									auto srcfields = src->getAllFields();

									uint outFieldIndex = 0;
									bool fieldFound = false;
									for (auto f : srcfields)
									{
										if (f == fieldSrc)
										{
											fieldFound = true;
											break;
										}
										outFieldIndex++;
									}

									if (fieldFound && moduleMap.find(outId) != moduleMap.end())
									{
										auto outBlock = moduleMap[outId];

										tinyxml2::XMLElement* moduleConnectionXml = doc.NewElement("Connection");
										moduleConnectionXml->SetAttribute("SourceId", ModuleIndices[outBlock->objectId()]);
										moduleConnectionXml->SetAttribute("From", outFieldIndex);
										moduleConnectionXml->SetAttribute("TargetId", ModuleIndices[m->objectId()]);
										moduleConnectionXml->SetAttribute("To", inFieldIndex);
										moduleConnectionsXml->InsertEndChild(moduleConnectionXml);
									}
								}
								else {//每个模块里面第一个
									Node* src = dynamic_cast<Node*>(parSrc);

									if (src != nullptr)
									{
										bool fieldFound = false;
										auto n = src->getModuleList().size();
<<<<<<< HEAD
										for (auto f : fieldsOut)
=======
										FieldTypeEnum type;
										auto fields = src->getAllFields();
										int nodeExportIndex = 0;
										for (auto f : fields)
>>>>>>> public
										{
											if (f == fieldSrc)
											{
												fieldFound = true;
												type = f->getFieldType();
												break;
											}
<<<<<<< HEAD
											outFieldIndex++;
=======
											nodeExportIndex++;
>>>>>>> public
										}

										if (fieldFound)
										{
											tinyxml2::XMLElement* moduleConnectionXml = doc.NewElement("Connection");
											moduleConnectionXml->SetAttribute("SourceId", -1 * int((indices[src->objectId()]) + 1));
											moduleConnectionXml->SetAttribute("From", nodeExportIndex);
											moduleConnectionXml->SetAttribute("TargetId", ModuleIndices[m->objectId()]);
											moduleConnectionXml->SetAttribute("To", inFieldIndex);

											moduleConnectionsXml->InsertEndChild(moduleConnectionXml);
										}
									}
								}
							}
						}
					}
				}

<<<<<<< HEAD
				indices.clear();
				};

			savePipeline(node->animationPipeline(), "Simulation");

			// 			//Insert graphics pipeline
			// 			tinyxml2::XMLElement* graphicsPipelineXml = doc.NewElement("Rendering");
			// 			nodeXml->InsertEndChild(graphicsPipelineXml);
			// 
			// 			auto graphicsPipeline = node->graphicsPipeline();
			// 			for each (auto m in graphicsPipeline->activeModules())
			// 			{
			// 				tinyxml2::XMLElement* moduleXml = doc.NewElement("Module");
			// 				moduleXml->SetAttribute("Class", m->getClassInfo()->getClassName().c_str());
			// 				moduleXml->SetAttribute("Coordinate", encodeVec2f(Vec2f(m->bx(), m->by())).c_str());
			// 				graphicsPipelineXml->InsertEndChild(moduleXml);
			// 			}

=======
				ModuleIndices.clear();
			};

			savePipeline(node->animationPipeline(), "Simulation");

>>>>>>> public
			savePipeline(node->graphicsPipeline(), "Rendering");


		}

		doc.SaveFile(filename.c_str());

		indices.clear();

		return true;
	}

	std::shared_ptr<Node> SceneLoaderXML::processNode(tinyxml2::XMLElement* nodeXML)
	{
		const char* name = nodeXML->Attribute("Class");
		if (!name)
			return nullptr;

		std::shared_ptr<Node> node(dynamic_cast<Node*>(Object::createObject(name)));
		if (node == nullptr)
		{
			std::cout << name << " does not exist! " << std::endl;
			return nullptr;
		}

		const char* coordStr = nodeXML->Attribute("Coordinate");

		Vec2f coord = decodeVec2f(std::string(coordStr));

		node->setBlockCoord(coord.x, coord.y);

		const char* nodeName = nodeXML->Attribute("Name");
		if (nodeName)
			node->setName(nodeName);

		std::map<std::string, FBase*> str2Field;
		auto& params = node->getParameters();
		for (auto var : params) {
			str2Field[var->getObjectName()] = var;
		}
		tinyxml2::XMLElement* varsXmls = nodeXML->FirstChildElement("Variables");
		tinyxml2::XMLElement* fieldXml = varsXmls->FirstChildElement("Field");
		while (fieldXml)
		{
			std::string name = fieldXml->Attribute("Name");
			std::string str = fieldXml->GetText() == nullptr ? "" : fieldXml->GetText();
			str2Field[name]->deserialize(str);

			fieldXml = fieldXml->NextSiblingElement("Field");
		}
		str2Field.clear();

		std::vector<ConnectionInfo> infoVec;

		/**
		 * Construct the node connections
		 */
		tinyxml2::XMLElement* cnnXmls = nodeXML->FirstChildElement("Connections");
		tinyxml2::XMLElement* connectionXml = cnnXmls->FirstChildElement("Connection");

		while (connectionXml)
		{
			ConnectionInfo info;

			info.src = atoi(connectionXml->Attribute("SourceId"));
			info.dst = atoi(connectionXml->Attribute("TargetId"));

			info.id0 = atoi(connectionXml->Attribute("From"));
			info.id1 = atoi(connectionXml->Attribute("To"));

<<<<<<< HEAD
			infoVec.push_back(info);


=======
			auto sourceModuleId = connectionXml->Attribute("SourceModuleId");
			auto sourceModulePiplineId = connectionXml->Attribute("SourceModulePipline");
			if (sourceModuleId != nullptr && sourceModulePiplineId != nullptr)
			{
				info.srcModuleId = atoi(connectionXml->Attribute("SourceModuleId"));
				info.srcModulePipline = atoi(connectionXml->Attribute("SourceModulePipline"));

			}

			infoVec.push_back(info);

>>>>>>> public
			connectionXml = connectionXml->NextSiblingElement("Connection");
		}

		mConnectionInfo.push_back(infoVec);

		infoVec.clear();

		auto& fields = node->getAllFields();
<<<<<<< HEAD
		std::vector<FBase*> states;
		for (auto field : fields)
		{
			if (field->getFieldType() == dyno::FieldTypeEnum::State)
			{
				states.push_back(field);
			}
			else if (field->getFieldType() == dyno::FieldTypeEnum::In)
			{

				states.push_back(field);

			}
		}
		/**
		 * Construct the animation pipeline
		 */
		std::vector<Module*> animationModules;
		node->animationPipeline()->clear();
		tinyxml2::XMLElement* animationPipelineXml = nodeXML->FirstChildElement("Simulation");
		tinyxml2::XMLElement* animationModuleXml = animationPipelineXml->FirstChildElement("Module");

		auto processModule = [&](tinyxml2::XMLElement* moduleXml, std::shared_ptr<Pipeline> pipeline, std::vector<Module*>& modules) {
			const char* name = moduleXml->Attribute("Class");
			if (name) {
				std::shared_ptr<Module> module(dynamic_cast<Module*>(Object::createObject(name)));
				if (module != nullptr) {
					pipeline->pushModule(module);
					modules.push_back(module.get());
=======
>>>>>>> public


<<<<<<< HEAD
					tinyxml2::XMLElement* varsXml = moduleXml->FirstChildElement("Variables");
					tinyxml2::XMLElement* varXml = varsXml->FirstChildElement("Field");

					while (varXml)
					{
						std::string name = varXml->Attribute("Name");
						std::string str = varXml->GetText();
						str2Field[name]->deserialize(str);

						varXml = varXml->NextSiblingElement("Field");
					}

					str2Field.clear();
				}
			}
			};

		while (animationModuleXml)
		{
			processModule(animationModuleXml, node->animationPipeline(), animationModules);
			// 			const char* name = animationModuleXml->Attribute("Class");
			// 			if (name) {
			// 				std::shared_ptr<Module> module(dynamic_cast<Module*>(Object::createObject(name)));
			// 				if (module != nullptr) {
			// 					node->animationPipeline()->pushModule(module);
			// 					animationModules.push_back(module.get());
			// 				}
			// 			}

			animationModuleXml = animationModuleXml->NextSiblingElement("Module");
		}
		tinyxml2::XMLElement* animationConnectionsXml = animationPipelineXml->FirstChildElement("Connections");
		tinyxml2::XMLElement* animationConnectionXml = animationConnectionsXml->FirstChildElement("Connection");
		while (animationConnectionXml)
		{
			int src = atoi(animationConnectionXml->Attribute("SourceId"));
			int dst = atoi(animationConnectionXml->Attribute("TargetId"));

			int id0 = atoi(animationConnectionXml->Attribute("From"));
			int id1 = atoi(animationConnectionXml->Attribute("To"));

			FBase* fout = src == -1 ? states[id0] : animationModules[src]->getOutputFields()[id0];
			FBase* fin = animationModules[dst]->getInputFields()[id1];

			fout->connect(fin);

			animationConnectionXml = animationConnectionXml->NextSiblingElement("Connection");
		}
		animationModules.clear();

		/**
		 * Construct the graphics pipeline
		 */
		std::vector<Module*> renderingModules;
		node->graphicsPipeline()->clear();
		tinyxml2::XMLElement* renderingPipelineXml = nodeXML->FirstChildElement("Rendering");
		tinyxml2::XMLElement* renderingModuleXml = renderingPipelineXml->FirstChildElement("Module");
		while (renderingModuleXml)
		{
			processModule(renderingModuleXml, node->graphicsPipeline(), renderingModules);
			// 			const char* name = renderingModuleXml->Attribute("Class");
			// 			if (name) {
			// 				std::shared_ptr<Module> module(dynamic_cast<Module*>(Object::createObject(name)));
			// 				if (module != nullptr) {
			// 					node->graphicsPipeline()->pushModule(module);
			// 					renderingModules.push_back(module.get());
			// 				}
			// 			}

			renderingModuleXml = renderingModuleXml->NextSiblingElement("Module");
		}

		tinyxml2::XMLElement* renderingConnectionsXml = renderingPipelineXml->FirstChildElement("Connections");
		tinyxml2::XMLElement* renderingConnectionXml = renderingConnectionsXml->FirstChildElement("Connection");
		while (renderingConnectionXml)
		{
			int src = atoi(renderingConnectionXml->Attribute("SourceId"));
			int dst = atoi(renderingConnectionXml->Attribute("TargetId"));

			int id0 = atoi(renderingConnectionXml->Attribute("From"));
			int id1 = atoi(renderingConnectionXml->Attribute("To"));

			if (id0 < states.size())
			{
				FBase* fout = src == -1 ? states[id0] : renderingModules[src]->getOutputFields()[id0];
				FBase* fin = renderingModules[dst]->getInputFields()[id1];

				fout->connect(fin);
			}


			renderingConnectionXml = renderingConnectionXml->NextSiblingElement("Connection");
		}
		renderingModules.clear();
=======
>>>>>>> public

		return node;
	}

<<<<<<< HEAD
	std::shared_ptr<Module> SceneLoaderXML::processModule(tinyxml2::XMLElement* moduleXML)
	{
		const char* className = moduleXML->Attribute("class");
		if (!className)
			return nullptr;

		const char* dataType = moduleXML->Attribute("datatype");
		std::string templateClass = std::string(className);
		if (dataType)
		{
			templateClass += std::string("<") + std::string(dataType) + std::string(">");
		}
		std::shared_ptr<Module> module(dynamic_cast<Module*>(Object::createObject(templateClass)));

		return module;
	}

	bool SceneLoaderXML::addModule(std::shared_ptr<Node> node, std::shared_ptr<Module> module)
	{
		if (module->getModuleType() == "MechanicalState")
		{
			return true;
		}

		return false;
	}

	std::vector<std::string> SceneLoaderXML::split(std::string str, std::string pattern)
	{
		std::string::size_type pos;
		std::vector<std::string> result;
		str += pattern;
		size_t size = (uint)str.size();

		for (size_t i = 0; i < size; i++)
		{
			pos = str.find(pattern, i);
			if (pos < size)
			{
				std::string s = str.substr(i, pos - i);
				result.push_back(s);
				i = pos + pattern.size() - 1;
			}
		}
		return result;
	}
=======
>>>>>>> public

	std::string SceneLoaderXML::encodeVec3f(const Vec3f v)
	{
		std::stringstream ss;

		ss << v.x << " " << v.y << " " << v.z;

		return ss.str();
	}

	std::string SceneLoaderXML::encodeVec2f(const Vec2f v)
	{
		std::stringstream ss;

		ss << v.x << " " << v.y;

		return ss.str();
	}

	Vec3f SceneLoaderXML::decodeVec3f(const std::string str)
	{
		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		float x = std::stof(substr.c_str());

		ss >> substr;
		float y = std::stof(substr.c_str());

		ss >> substr;
		float z = std::stof(substr.c_str());

		return Vec3f(x, y, z);
	}

	Vec2f SceneLoaderXML::decodeVec2f(const std::string str)
	{
		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		float x = std::stof(substr.c_str());

		ss >> substr;
		float y = std::stof(substr.c_str());

		return Vec2f(x, y);
	}

	bool SceneLoaderXML::canLoadFileByExtension(const std::string extension)
	{
		std::string str = extension;
		std::transform(str.begin(), str.end(), str.begin(), ::tolower);
		return (extension == "xml");
	}

	bool SceneLoaderXML::ConstructNodePipeline(std::shared_ptr<Node> node, tinyxml2::XMLElement* nodeXML, std::vector<std::shared_ptr<Node>> nodes)
	{
		bool anim = ConstructPipeline(node, node->animationPipeline(), nodeXML, nodes);
		bool graphic = ConstructPipeline(node, node->graphicsPipeline(), nodeXML, nodes);

		return anim && graphic;
	}

	bool SceneLoaderXML::ConstructPipeline(std::shared_ptr<Node> node, std::shared_ptr<Pipeline> pipeline, tinyxml2::XMLElement* nodeXML, std::vector<std::shared_ptr<Node>> nodes)
	{
		std::string pipelineName;

		if (dynamic_cast<AnimationPipeline*>(pipeline.get()))
			pipelineName = "Simulation";
		else if (dynamic_cast<GraphicsPipeline*>(pipeline.get()))
			pipelineName = "Rendering";
		else
			return false;


		std::vector<Module*> pipelineModules;
		pipeline->clear();
		tinyxml2::XMLElement* PipelineXml = nodeXML->FirstChildElement(pipelineName.c_str());
		tinyxml2::XMLElement* ModuleXml = PipelineXml->FirstChildElement("Module");

		

		while (ModuleXml)
		{
			processModule(ModuleXml, pipeline, pipelineModules);


			ModuleXml = ModuleXml->NextSiblingElement("Module");
		}
		tinyxml2::XMLElement* ConnectionsXml = PipelineXml->FirstChildElement("Connections");
		tinyxml2::XMLElement* ConnectXml = ConnectionsXml->FirstChildElement("Connection");
		while (ConnectXml)
		{
			int src = atoi(ConnectXml->Attribute("SourceId"));
			int dst = atoi(ConnectXml->Attribute("TargetId"));

			int id0 = atoi(ConnectXml->Attribute("From"));
			int id1 = atoi(ConnectXml->Attribute("To"));
			FBase* fout;
			if (src < 0) 
			{
				auto& fields = nodes[std::abs(src) - 1]->getAllFields();
				fout = fields[id0];
			}
			else 
			{
				fout = pipelineModules[src]->getAllFields()[id0];
			}


			FBase* fin = pipelineModules[dst]->getAllFields()[id1];

			fout->connect(fin);

			ConnectXml = ConnectXml->NextSiblingElement("Connection");
		}
		pipelineModules.clear();
		return true;
	}

	void SceneLoaderXML::processModule(tinyxml2::XMLElement* moduleXml, std::shared_ptr<Pipeline> pipeline, std::vector<Module*>& modules)
	{
		const char* name = moduleXml->Attribute("Class");
		if (name) {
			std::shared_ptr<Module> module(dynamic_cast<Module*>(Object::createObject(name)));
			if (module != nullptr) {
				pipeline->pushModule(module);
				modules.push_back(module.get());

				std::map<std::string, FBase*> str2Field;
				auto& params = module->getParameters();
				for (auto var : params) {
					str2Field[var->getObjectName()] = var;
				}

				tinyxml2::XMLElement* varsXml = moduleXml->FirstChildElement("Variables");
				tinyxml2::XMLElement* varXml = varsXml->FirstChildElement("Field");

				while (varXml)
				{
					std::string name = varXml->Attribute("Name");
					std::string str = varXml->GetText();
					str2Field[name]->deserialize(str);

					varXml = varXml->NextSiblingElement("Field");
				}

				str2Field.clear();
			}
		}
	}

}
