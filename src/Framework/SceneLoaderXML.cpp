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

		scn->setUpperBound(decodeVec3f(root->Attribute("UpperBound")));
		scn->setLowerBound(decodeVec3f(root->Attribute("LowerBound")));

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

		scn->markQueueUpdateRequired();

		return scn;
	}

	bool SceneLoaderXML::save(std::shared_ptr<SceneGraph> scn, const std::string filename)
	{
		tinyxml2::XMLDocument doc;

		// 		const char* declaration = "<?xml version=\"0.6.0\" encoding=\"UTF-8\">";
		// 		doc.Parse(declaration);

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
			nodeXml->SetAttribute("Dt",node->getDt());
			nodeXml->SetAttribute("AutoSync", node->isAutoSync());
			nodeXml->SetAttribute("AutoHidden", node->isAutoHidden());
			nodeXml->SetAttribute("PhysicsEnabled", node->isActive());
			nodeXml->SetAttribute("RenderingEnabled", node->isVisible());

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
			}

			for (auto s : node->getOutputFields())
			{
				auto fieldSrc = s->getSource();
				if (fieldSrc != nullptr)
				{
					auto parSrc = fieldSrc->parent();
					int ToIndex = -1;
					auto fields = node->getAllFields();
					for (size_t i = 0; i < fields.size(); i++)
					{
						auto t = fields[i];
						if (t == s)
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
										FieldTypeEnum type;
										auto fields = src->getAllFields();
										int nodeExportIndex = 0;
										for (auto f : fields)
										{
											if (f == fieldSrc)
											{
												fieldFound = true;
												type = f->getFieldType();
												break;
											}
											nodeExportIndex++;
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

				ModuleIndices.clear();
			};

			savePipeline(node->animationPipeline(), "Simulation");

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
		
		const char* dtStr = nodeXML->Attribute("Dt");
		node->setDt(std::stod(dtStr));

		const char* syncStr = nodeXML->Attribute("AutoSync");
		node->setAutoSync(std::string(syncStr) == "false" ? false : true);

		const char* hiddenStr = nodeXML->Attribute("AutoHidden");
		node->setAutoHidden(std::string(hiddenStr) == "false" ? false : true);

		const char* activeStr = nodeXML->Attribute("PhysicsEnabled");
		node->setActive(std::string(activeStr) == "false" ? false : true);

		const char* visibleStr = nodeXML->Attribute("RenderingEnabled");
		node->setVisible(std::string(visibleStr) == "false" ? false : true);

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

			auto sourceModuleId = connectionXml->Attribute("SourceModuleId");
			auto sourceModulePiplineId = connectionXml->Attribute("SourceModulePipline");
			if (sourceModuleId != nullptr && sourceModulePiplineId != nullptr)
			{
				info.srcModuleId = atoi(connectionXml->Attribute("SourceModuleId"));
				info.srcModulePipline = atoi(connectionXml->Attribute("SourceModulePipline"));

			}

			infoVec.push_back(info);

			connectionXml = connectionXml->NextSiblingElement("Connection");
		}

		mConnectionInfo.push_back(infoVec);

		infoVec.clear();

		auto& fields = node->getAllFields();



		return node;
	}


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
