#include "ConvertToTextureMesh.h"

#include "GLPhotorealisticRender.h"
#include "Primitive/Primitive3D.h"
#include "MaterialFunc.h"


namespace dyno
{
	template< typename Triangle, typename Coord >
	__global__ void updateNormal(
		DArray<Triangle> Tris,
		DArray<Coord> Pts,
		DArray<Coord> Normals
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Tris.size()) return;

		Vec3f a;
		Vec3f b;

		//Tri[0]
		a = Pts[Tris[pId][0]] - Pts[Tris[pId][1]];
		b = Pts[Tris[pId][2]] - Pts[Tris[pId][0]];
		Normals[pId * 3] = (a.normalize()).cross(b.normalize());

		//Tri[1]
		a = Pts[Tris[pId][1]] - Pts[Tris[pId][2]];
		b = Pts[Tris[pId][0]] - Pts[Tris[pId][1]];
		Normals[pId * 3 + 1] = (a.normalize()).cross(b.normalize());

		//Tri[2]
		a = Pts[Tris[pId][2]] - Pts[Tris[pId][0]];
		b = Pts[Tris[pId][1]] - Pts[Tris[pId][2]];
		Normals[pId * 3 + 2] = (a.normalize()).cross(b.normalize());

	}

	template< typename Triangle, typename Coord, typename Vec2f>
	__global__ void updateTexCoord(
		DArray<Triangle> Tris,
		DArray<Coord> Pts,
		DArray<Vec2f> TexCoords
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Tris.size()) return;

		//Tri[0]
		TexCoords[3 * pId] = Vec2f(Pts[Tris[pId][0]][0], Pts[Tris[pId][0]][1]);

		//Tri[1]
		TexCoords[3 * pId + 1] = Vec2f(Pts[Tris[pId][1]][0], Pts[Tris[pId][1]][1]);

		//Tri[2]
		TexCoords[3 * pId + 2] = Vec2f(Pts[Tris[pId][2]][0], Pts[Tris[pId][2]][1]);

	}

	template< typename Triangle, typename Coord, typename Vec2f>
	__global__ void updateTexCoordByNormal(
		DArray<Triangle> Tris,
		DArray<Coord> Pts,
		DArray<Vec2f> TexCoords
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Tris.size()) return;

		Vec3f a;
		Vec3f b;

		//Tri[0]
		a = Pts[Tris[pId][0]] - Pts[Tris[pId][1]];
		b = Pts[Tris[pId][2]] - Pts[Tris[pId][0]];
		Vec3f faceNormal = (a.normalize()).cross(b.normalize()).normalize();

		//ÅÐ¶Ï·½Ïò
		Vec3f x = Vec3f(1, 0, 0);
		Vec3f y = Vec3f(0, 1, 0);
		Vec3f z = Vec3f(0, 0, 1);

		int mode = -1;//x = 0,y = 1,z =2;

		float xdot = faceNormal.dot(x);
		float ydot = faceNormal.dot(y);
		float zdot = faceNormal.dot(z);

		mode = abs(xdot) < abs(ydot) ? 0 : 1;
		float temp = abs(xdot) < abs(ydot) ? abs(xdot) : abs(ydot);
		mode = temp < abs(zdot) ? mode : 2;

		printf("%d - %f,%f,%f\n",mode,abs(xdot), abs(ydot), abs(zdot));
		//Ó³Éä
		float scale = 5; 
		if (mode == 0) 
		{
			//Tri[0]
			TexCoords[3 * pId] = Vec2f(Pts[Tris[pId][0]][0], Pts[Tris[pId][0]][2]) * scale;
			//Tri[1]
			TexCoords[3 * pId + 1] = Vec2f(Pts[Tris[pId][1]][0], Pts[Tris[pId][1]][2]) * scale;
			//Tri[2]
			TexCoords[3 * pId + 2] = Vec2f(Pts[Tris[pId][2]][0], Pts[Tris[pId][2]][2]) * scale;
		}
		if (mode == 1) 
		{
			//Tri[0]
			TexCoords[3 * pId] = Vec2f(Pts[Tris[pId][0]][0], Pts[Tris[pId][0]][1]) * scale;
			//Tri[1]
			TexCoords[3 * pId + 1] = Vec2f(Pts[Tris[pId][1]][0], Pts[Tris[pId][1]][1]) * scale;
			//Tri[2]
			TexCoords[3 * pId + 2] = Vec2f(Pts[Tris[pId][2]][0], Pts[Tris[pId][2]][1]) * scale;
			
		}
		if (mode == 2) 
		{
			//Tri[0]
			TexCoords[3 * pId] = Vec2f(Pts[Tris[pId][0]][0], Pts[Tris[pId][0]][1]) * scale;
			//Tri[1]
			TexCoords[3 * pId + 1] = Vec2f(Pts[Tris[pId][1]][0], Pts[Tris[pId][1]][1]) * scale;
			//Tri[2]
			TexCoords[3 * pId + 2] = Vec2f(Pts[Tris[pId][2]][0], Pts[Tris[pId][2]][1]) * scale;
		}


	}

	template< typename Coord, typename Triangle>
	__global__ void convertTriangleSetToTexturemesh(
		DArray<Coord> srcPts,
		DArray<Triangle> srcTris,
		DArray<Coord> targetPts,
		DArray<Triangle> targetTris
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= srcTris.size()) return;

		//Tri[0]
		targetPts[3 * pId] = srcPts[srcTris[pId][0]];
		targetTris[pId][0] = 3 * pId;

		//Tri[1]
		targetPts[3 * pId + 1] = srcPts[srcTris[pId][1]];
		targetTris[pId][1] = 3 * pId + 1;

		//Tri[2]		
		targetPts[3 * pId + 2] = srcPts[srcTris[pId][2]];
		targetTris[pId][2] = 3 * pId + 2;
	}

	template<typename TDataType>
	ConvertToTextureMesh<TDataType>::ConvertToTextureMesh()
		: Node()
	{
		this->stateTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());

		auto render = std::make_shared<GLPhotorealisticRender>();
		this->stateTextureMesh()->connect(render->inTextureMesh());
		this->graphicsPipeline()->pushModule(render);

		this->stateTextureMesh()->promoteOuput();
	}

	template<typename TDataType>
	void ConvertToTextureMesh<TDataType>::resetStates()
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->inTopology()->getDataPtr());
		if (triSet == NULL)
			return;

		std::shared_ptr<TextureMesh> texMesh = this->stateTextureMesh()->getDataPtr();
		texMesh->shapes().resize(1);
		texMesh->materials().resize(1);


		auto SourceTriangles = triSet->getTriangles();
		auto SourcePoints = triSet->getPoints();

		texMesh->shapes()[0] = std::make_shared<Shape>();
		texMesh->shapes()[0]->vertexIndex.resize(SourceTriangles.size());
		
	

		texMesh->vertices().resize(SourceTriangles.size() * 3);

		auto& TargetTriangles = texMesh->shapes()[0]->vertexIndex;
		auto& TargetPoints = texMesh->vertices();

		// ShapeIndex and rebuild Points;
		cuExecute(SourceTriangles.size(),
			convertTriangleSetToTexturemesh,
			SourcePoints,
			SourceTriangles,
			TargetPoints,
			TargetTriangles
		);


		// Normals
		auto& Normals = texMesh->normals();
		Normals.resize(TargetPoints.size());


		cuExecute(TargetTriangles.size(),
			updateNormal,
			TargetTriangles,
			TargetPoints,
			Normals
		);

		auto& TexCoords = texMesh->texCoords();
		TexCoords.resize(TargetPoints.size());

		cuExecute(TargetTriangles.size(),
			updateTexCoordByNormal,//updateTexCoord
			TargetTriangles,
			TargetPoints,
			TexCoords
		);

		// ShapeIndex ;
		texMesh->shapes()[0]->normalIndex.assign(TargetTriangles);
		texMesh->shapes()[0]->texCoordIndex.assign(TargetTriangles);


		// Move To Center, use Transform
		Reduction<Coord> reduceBounding;

		auto& bounding = texMesh->shapes()[0]->boundingBox;
		Coord lo = reduceBounding.minimum(SourcePoints.begin(), SourcePoints.size());
		Coord hi = reduceBounding.maximum(SourcePoints.begin(), SourcePoints.size());

		bounding.v0 = lo;
		bounding.v1 = hi;

		texMesh->shapes()[0]->boundingTransform.translation() = (lo + hi) / 2;
		// Assign Material;

		texMesh->materials()[0] = std::make_shared<Material>();
		texMesh->shapes()[0]->material = texMesh->materials()[0];

		//set Material
		dyno::CArray2D<dyno::Vec4f> texture(1, 1);

		auto diffusePath = this->varDiffuseTexture()->getValue().string();
		if (diffusePath.size())
		{
			if (loadTexture(diffusePath.c_str(), texture))
				texMesh->shapes()[0]->material->texColor.assign(texture);
		}

		auto bumpPath = this->varNormalTexture()->getValue().string();
		if (bumpPath.size()) 
		{
			if (loadTexture(bumpPath.c_str(), texture))
				texMesh->shapes()[0]->material->texBump.assign(texture);
		}

		//
		texMesh->shapeIds().assign(std::vector<uint>(TargetPoints.size(), 0));
	}

	DEFINE_CLASS(ConvertToTextureMesh);
}