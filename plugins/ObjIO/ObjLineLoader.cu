#include "ObjLineLoader.h"

#include "Topology/TriangleSet.h"
#include <iostream>
#include <sys/stat.h>
#include "tinyobjloader/tiny_obj_loader.h"
#include "helpers/tinyobj_helper.h"
#include "tinyobjloader/tiny_obj_loader.h"
#include "GLWireframeVisualModule.h"
#include <fstream>
#include <sstream>
#include <cctype>
#include <cmath>
#include <iostream>


namespace dyno
{
	IMPLEMENT_TCLASS(ObjLine, TDataType)

		template<typename TDataType>
	ObjLine<TDataType>::ObjLine()
		: Node()
	{
		auto edgeSet = std::make_shared<EdgeSet<TDataType>>();
		this->varRadius()->setRange(0,10);

		this->stateEdgeSet()->setDataPtr(edgeSet);

		auto wireRender = std::make_shared<GLWireframeVisualModule>();
		
		wireRender->setVisible(true);
		wireRender->setColor(Color::Green());

		this->stateEdgeSet()->connect(wireRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireRender);

	}

	template<typename TDataType>
	void ObjLine<TDataType>::resetStates()
	{
		auto edgeSet = this->stateEdgeSet()->getDataPtr();

		std::string filename = this->varFileName()->constDataPtr()->string();

		loadObj(*edgeSet, filename);
		edgeSet->scale(this->varScale()->getData());
		edgeSet->translate(this->varLocation()->getData());
		edgeSet->rotate(this->varRotation()->getData() * M_PI / 180);

		Node::resetStates();
		
		initPos.assign(edgeSet->getPoints());

	}


	template<typename TDataType>
	void ObjLine<TDataType>::loadObj(EdgeSet<TDataType>& edgeSet, std::string filename)
	{

		std::vector<Coord> vertList;
		std::vector<TopologyModule::Edge> edgeList;
		parseOBJ(filename, vertList, edgeList );

		edgeSet.setPoints(vertList);
		edgeSet.setEdges(edgeList);
		edgeSet.update();
	}

	template<typename TDataType>
	void ObjLine<TDataType>::parseOBJ(const std::string& filename, std::vector<Vec3f>& vertices, std::vector<TopologyModule::Edge>& edges)
	{
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        std::string line;
        std::string currentLine = "";
        bool continueLine = false;

        while (std::getline(file, line)) {
            // 
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            if (continueLine) {
                currentLine += " " + trim(line);
                continueLine = false;
            }
            else {
                currentLine = trim(line);
            }

            if (!currentLine.empty() && currentLine.back() == '\\') {
                currentLine.pop_back(); 
                currentLine = trim(currentLine); 
                continueLine = true;
                continue;
            }

            if (currentLine.empty() || currentLine[0] == '#') {
                continue;
            }

            std::istringstream iss(currentLine);
            std::string prefix;
            iss >> prefix;

            // parsePoint
            if (prefix == "v") {
                float x, y, z;
                if (iss >> x >> y >> z) {
                    vertices.emplace_back(x, y, z);
                }
                else {
                    std::cerr << "Warning: Invalid vertex line: " << currentLine << std::endl;
                }
            }
            // parseLine
            else if (prefix == "l") {
                std::vector<int> indices;
                int idx;

                // parseIndices
                while (iss >> idx) {
                    indices.push_back(idx);
                }

                // Convert Indices
                for (size_t i = 0; i < indices.size(); i++) {
                    // <0
                    if (indices[i] < 0) {
                        indices[i] = vertices.size() + indices[i];
                    }
                    // >0
                    else {
                        indices[i] = indices[i] - 1;
                    }

                    if (indices[i] < 0 || indices[i] >= static_cast<int>(vertices.size())) {
                        std::cerr << "Warning: Invalid index in line: " << currentLine << std::endl;
                        indices.clear();
                        break;
                    }
                }

                
                for (size_t i = 1; i < indices.size(); i++) {
                    edges.emplace_back(indices[i - 1], indices[i]);
                }
            }

            currentLine = "";
        }
	}


	DEFINE_CLASS(ObjLine);
}