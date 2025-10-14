
#include "Subdivide.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"

namespace dyno
{

    template<typename TDataType>
    void Subdivide<TDataType>::loopSubdivide(std::vector<Vec3f>& vertices, std::vector<TopologyModule::Triangle>& triangles) {
        Map<EKey, uint> middlePointIndexCache;
        std::vector<TopologyModule::Triangle> newTriangles;
        std::vector<Vec3f> newVertices;
        TriangleSet<DataType3f> triSet;

        newVertices = vertices;
        triSet.setPoints(vertices);
        triSet.setTriangles(triangles);
        triSet.update();
       
        CArray<TopologyModule::Edg2Tri> c_Edg2Tri;
        c_Edg2Tri.assign(triSet.edge2Triangle());
        CArray<TopologyModule::Tri2Edg> c_Tri2Edg;
        c_Tri2Edg.assign(triSet.triangle2Edge());
        CArray<TopologyModule::Edge> c_Edge;
        c_Edge.assign(triSet.edgeIndices());
        CArrayList<int> c_ver2edge;
        c_ver2edge.assign(triSet.vertex2Edge());

        Pair<EKey, uint>* pairs = new Pair<EKey, uint>[c_Edge.size()];
        middlePointIndexCache.reserve(pairs, c_Edge.size());

        auto getTriOtherPoint = [&](uint v0, uint v1, uint triId) {
            if (triangles[triId][0] != v0 && triangles[triId][0] != v1)
                return int(triangles[triId][0]);
            else if (triangles[triId][1] != v0 && triangles[triId][1] != v1)
                return int(triangles[triId][1]);
            else if (triangles[triId][2] != v0 && triangles[triId][2] != v1)
                return int(triangles[triId][2]);
            else
            {
                return int(-1);
            }
        };

        auto getMiddlePoint = [&](int eId) {
            uint v0 = c_Edge[eId][0];
            uint v1 = c_Edge[eId][1];

            uint tri0 = c_Edg2Tri[eId][0];
            uint tri1 = c_Edg2Tri[eId][1];

            int v3 = tri0 != -1 ? getTriOtherPoint(v0, v1, tri0) : -1;
            int v4 = tri1 != -1 ? getTriOtherPoint(v0, v1, tri1) : -1;

            Vec3f p0 = vertices[v0];
            Vec3f p1 = vertices[v1];

            Vec3f p3 = vertices[v3];
            Vec3f p4 = vertices[v4];

            Vec3f newV;
            if (v3 != -1 && v4 != -1)
            {
                newV = 3.0f / 8.0f * (p0 + p1) + 1.0f / 8.0f * (p3 + p4);
            }
            else
                newV = 1.0f / 2.0f * (p0 + p1);

            EKey edgeKey = EKey(v0, v1);
            if (middlePointIndexCache.find(edgeKey) == nullptr)
                middlePointIndexCache.insert(Pair<EKey, uint>(edgeKey, vertices.size() + eId));

            newVertices.push_back(newV);

        };

        for (size_t i = 0; i < c_Edge.size(); i++)
        {
            //printf("\n********  %d  ******\n", i);

            getMiddlePoint(i);
        }


        auto updateOldPoints = [&]() {

            auto getEdgeOtherPoint = [&](int vId, TopologyModule::Edge edge) {
                if (edge[0] != vId && edge[1] != vId)
                {
                    printf("Error getEdgeOtherPoint!!!!\ninput vId: %d\nedge vId: %d, %d \n", vId, edge[0], edge[1]);
                    return -1;
                }
                else
                {
                    int targetV = edge[0] == vId ? edge[1] : edge[0];
                    //printf("getEdgeOtherPoint: \ninput V: %d\noutput V: %d \n", vId, targetV);
                    return targetV;
                }
            };



            for (size_t i = 0; i < c_ver2edge.size(); i++)
            {
                uint v0 = i;
                auto edgeList = c_ver2edge[v0];

                Vec3f sum = Vec3f(0);
                int n = edgeList.size();
                Vec3f p0 = vertices[v0];
                Vec3f newP;


                if (n > 2)
                {
                    for (size_t j = 0; j < n; j++)
                    {
                        uint edgeId = edgeList[j];
                        int connectionV = getEdgeOtherPoint(v0, c_Edge[edgeId]);
                        if (connectionV != -1)
                            sum += vertices[connectionV];
                    }

                    //double beta = 1 / double(n) * (5.0f / 8.0f - (3.0f / 8.0f + 1.0f / 4.0f * cos(2 * M_PI / double(n))));
                    double beta = n == 3 ? (double)3 / (double)16 : (double)3 / ((double)8 * (double)n);


                    //newP = (1 - double(n) * beta) * p0 + beta * sum;
                    newP = (1 - double(n) * beta) * p0 + beta * sum;
                }
                else if (n == 2)
                {
                    Vec3f p1 = vertices[getEdgeOtherPoint(v0, c_Edge[edgeList[0]])];
                    Vec3f p2 = vertices[getEdgeOtherPoint(v0, c_Edge[edgeList[1]])];
                    newP = 3 / 4 * p0 + 1.0f / 8.0f * p1 + 1.0f / 8.0f + p2;
                }
                newVertices[v0] = newP;
            }

        };

        updateOldPoints();

        vertices = newVertices;

        newTriangles.resize(4 * triangles.size());
        for (size_t i = 0; i < triangles.size(); i++)
        {
            int v01 = middlePointIndexCache.find(EKey(triangles[i][0], triangles[i][1])) != middlePointIndexCache.end() ? (middlePointIndexCache.find(EKey(triangles[i][0], triangles[i][1]))->second) : -1;
            int v12 = middlePointIndexCache.find(EKey(triangles[i][0], triangles[i][1])) != middlePointIndexCache.end() ? (middlePointIndexCache.find(EKey(triangles[i][1], triangles[i][2]))->second) : -1;
            int v20 = middlePointIndexCache.find(EKey(triangles[i][0], triangles[i][1])) != middlePointIndexCache.end() ? (middlePointIndexCache.find(EKey(triangles[i][2], triangles[i][0]))->second) : -1;

            int v0 = triangles[i][0];
            int v1 = triangles[i][1];
            int v2 = triangles[i][2];

            newTriangles[4 * i] = TopologyModule::Triangle(v01, v1, v12);
            newTriangles[4 * i + 1] = TopologyModule::Triangle(v12, v2, v20);
            newTriangles[4 * i + 2] = TopologyModule::Triangle(v20, v0, v01);
            newTriangles[4 * i + 3] = TopologyModule::Triangle(v01, v12, v20);
        }
        triangles = newTriangles;
    }


    template<typename TDataType>
    Subdivide<TDataType>::Subdivide()
        : ParametricModel<TDataType>()
    {
        this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

        this->varStep()->setRange(0, 6);

        auto callback = std::make_shared<FCallBackFunc>(std::bind(&Subdivide<TDataType>::varChanged, this));
        this->varStep()->attach(callback);

        auto tsRender = std::make_shared<GLSurfaceVisualModule>();
        this->stateTriangleSet()->connect(tsRender->inTriangleSet());
        this->graphicsPipeline()->pushModule(tsRender);


        auto esRender = std::make_shared<GLWireframeVisualModule>();
        esRender->varBaseColor()->setValue(Color(0, 0, 0));
        this->stateTriangleSet()->connect(esRender->inEdgeSet());
        this->graphicsPipeline()->pushModule(esRender);

        this->stateTriangleSet()->promoteOuput();
    }


    template<typename TDataType>
    void Subdivide<TDataType>::resetStates()
    {
        this->varChanged();
    }


    template<typename TDataType>
    void Subdivide<TDataType>::varChanged()
    {
        auto inTri = this->inInTriangleSet()->constDataPtr();

        std::vector<Vec3f> vts;
        std::vector<TopologyModule::Triangle> tris;

        CArray<Vec3f> c_v;
        CArray<TopologyModule::Triangle> c_t;

        c_v.assign(inTri->getPoints());
        c_t.assign(inTri->triangleIndices());

        for (size_t i = 0; i < c_v.size(); i++)
        {
            vts.push_back(c_v[i]);
        }
        for (size_t i = 0; i < c_t.size(); i++)
        {
            tris.push_back(c_t[i]);
        }

        for (int i = 0; i < this->varStep()->getValue(); i++)
        {
            loopSubdivide(vts, tris);
        }

        this->stateTriangleSet()->getDataPtr()->setPoints(vts);
        this->stateTriangleSet()->getDataPtr()->setTriangles(tris);
        this->stateTriangleSet()->getDataPtr()->update();
    }


    DEFINE_CLASS(Subdivide);

}