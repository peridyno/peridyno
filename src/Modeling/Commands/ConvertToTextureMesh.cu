#include "ConvertToTextureMesh.h"

#include "GLPhotorealisticRender.h"
#include "Primitive/Primitive3D.h"
#include "MaterialFunc.h"
#include "GLSurfaceVisualModule.h"
#include "GLPointVisualModule.h"

namespace dyno
{
	template< typename Coord, typename Vec3f >
	__global__ void ToCenter(
		DArray<Coord> iniPos,
		DArray<Coord> finalPos,
		Vec3f translation
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= iniPos.size()) return;

		finalPos[pId] = iniPos[pId] - translation;
	}

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
		Normals[pId * 3] = (a.normalize()).cross(b.normalize()).normalize();

		//Tri[1]
		a = Pts[Tris[pId][1]] - Pts[Tris[pId][2]];
		b = Pts[Tris[pId][0]] - Pts[Tris[pId][1]];
		Normals[pId * 3 + 1] = (a.normalize()).cross(b.normalize()).normalize();

		//Tri[2]
		a = Pts[Tris[pId][2]] - Pts[Tris[pId][0]];
		b = Pts[Tris[pId][1]] - Pts[Tris[pId][2]];
		Normals[pId * 3 + 2] = (a.normalize()).cross(b.normalize()).normalize();

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

	template<typename Coord, typename Vec2f>
	__global__ void updateTexCoordByVertexNormal(
		DArray<Vec3f> Normals,
		DArray<Coord> Pts,
		DArray<Vec2f> TexCoords,
		Vec2f scale
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Normals.size()) return;

		int mode = -1;//x = 0,y = 1,z =2;

		float xdot = abs(Normals[pId].dot(Vec3f(1, 0, 0)));
		float ydot = abs(Normals[pId].dot(Vec3f(0, 1, 0)));
		float zdot = abs(Normals[pId].dot(Vec3f(0, 0, 1)));

		mode = xdot > ydot ? 0 : 1;
		float temp = xdot > ydot ? xdot : ydot;
		mode = temp > zdot ? mode : 2;

		//UV Tri-Plane Project
		if (mode == 0)
			TexCoords[pId] = Vec2f(Pts[pId][2] * scale[0] , Pts[pId][1] * scale[1]);
		if (mode == 1) 
			TexCoords[pId] = Vec2f(Pts[pId][2] * scale[0], Pts[pId][0] * scale[1]) ;
		if (mode == 2) 
			TexCoords[pId] = Vec2f(Pts[pId][0] * scale[0], Pts[pId][1] * scale[1]) ;
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

		//Tri[1]
		targetPts[3 * pId + 1] = srcPts[srcTris[pId][1]];

		//Tri[2]		
		targetPts[3 * pId + 2] = srcPts[srcTris[pId][2]];

		//Triangle_PointID
		targetTris[pId][0] = 3 * pId;
		targetTris[pId][1] = 3 * pId + 1;
		targetTris[pId][2] = 3 * pId + 2;
	}

	template<typename TDataType>
	ConvertToTextureMesh<TDataType>::ConvertToTextureMesh()
		: ModelEditing<TDataType>()
	{
		this->stateTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());

		auto render = std::make_shared<GLPhotorealisticRender>();
		this->stateTextureMesh()->connect(render->inTextureMesh());
		this->graphicsPipeline()->pushModule(render);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&ConvertToTextureMesh<TDataType>::varChanged, this));
		
		auto pointsTrianglesCallback = std::make_shared<FCallBackFunc>(std::bind(&ConvertToTextureMesh<TDataType>::createTextureMesh, this));
		auto materialCallback = std::make_shared<FCallBackFunc>(std::bind(&ConvertToTextureMesh<TDataType>::createMaterial, this));
		auto normalCallback = std::make_shared<FCallBackFunc>(std::bind(&ConvertToTextureMesh<TDataType>::createNormal, this));
		auto uvCallback = std::make_shared<FCallBackFunc>(std::bind(&ConvertToTextureMesh<TDataType>::createUV, this));
		auto moveCenterCallback = std::make_shared<FCallBackFunc>(std::bind(&ConvertToTextureMesh<TDataType>::moveToCenter, this));


		this->varDiffuseTexture()->attach(materialCallback);
		this->varNormalTexture()->attach(materialCallback);
		this->varUseBoundingTransform()->attach(moveCenterCallback);
		this->varUvScaleU()->attach(uvCallback);
		this->varUvScaleV()->attach(uvCallback);

		this->varUvScaleU()->setRange(0, 20);
		this->varUvScaleV()->setRange(0, 20);

		this->stateTextureMesh()->promoteOuput();
	}

	template<typename TDataType>
	void ConvertToTextureMesh<TDataType>::resetStates()
	{
		this->varChanged();
	}

	template<typename TDataType>
	void ConvertToTextureMesh<TDataType>::varChanged() 
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->inTopology()->getDataPtr());
		if (triSet == NULL)
			return;

		std::shared_ptr<TextureMesh> texMesh = this->stateTextureMesh()->getDataPtr();
		texMesh->shapes().resize(1);

		createTextureMesh();

		createNormal();	

		createUV();

		moveToCenter();

		createMaterial();

		auto& TargetPoints = texMesh->meshDataPtr()->vertices();
		texMesh->meshDataPtr()->shapeIds().assign(std::vector<uint>(TargetPoints.size(), 0));
	
	}

	template<typename TDataType>
	void ConvertToTextureMesh<TDataType>::createTextureMesh()
	{
		std::shared_ptr<TextureMesh> texMesh = this->stateTextureMesh()->getDataPtr();
		//Points
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->inTopology()->getDataPtr());
		if (triSet == NULL)
			return;

		auto SourceTriangles = triSet->triangleIndices();
		auto SourcePoints = triSet->getPoints();

		texMesh->shapes()[0] = std::make_shared<Shape>();
		texMesh->shapes()[0]->vertexIndex.resize(SourceTriangles.size());

		texMesh->meshDataPtr()->vertices().resize(SourceTriangles.size() * 3);

		auto& TargetTriangles = texMesh->shapes()[0]->vertexIndex;
		auto& TargetPoints = texMesh->meshDataPtr()->vertices();

		// ShapeIndex and rebuild Points;
		cuExecute(SourceTriangles.size(),
			convertTriangleSetToTexturemesh,
			SourcePoints,
			SourceTriangles,
			TargetPoints,
			TargetTriangles
		);

	}

	template<typename TDataType>
	void ConvertToTextureMesh<TDataType>::createNormal()
	{
		std::shared_ptr<TextureMesh> texMesh = this->stateTextureMesh()->getDataPtr();
		auto& TargetTriangles = texMesh->shapes()[0]->vertexIndex;
		auto& TargetPoints = texMesh->meshDataPtr()->vertices();
		// Normals
		auto& Normals = texMesh->meshDataPtr()->normals();
		Normals.resize(TargetPoints.size());

		cuExecute(TargetTriangles.size(),
			updateNormal,
			TargetTriangles,
			TargetPoints,
			Normals
		);
		texMesh->shapes()[0]->normalIndex.assign(TargetTriangles);

	}

	template<typename TDataType>
	void ConvertToTextureMesh<TDataType>::createUV()
	{
		std::shared_ptr<TextureMesh> texMesh = this->stateTextureMesh()->getDataPtr();
		if (!bool(texMesh->shapes().size()))
			return;
		auto& TargetTriangles = texMesh->shapes()[0]->vertexIndex;
		auto& TargetPoints = texMesh->meshDataPtr()->vertices();
		auto& Normals = texMesh->meshDataPtr()->normals();

		auto& TexCoords = texMesh->meshDataPtr()->texCoords();
		TexCoords.resize(TargetPoints.size());
		//this->statePointColors()->resize(TargetPoints.size());
		//auto& colors = this->statePointColors()->getData();
		cuExecute(Normals.size(),
			updateTexCoordByVertexNormal,//updateTexCoord
			Normals,
			TargetPoints,
			TexCoords,
			Vec2f(this->varUvScaleU()->getValue(), this->varUvScaleV()->getValue())
		);
		texMesh->shapes()[0]->texCoordIndex.assign(TargetTriangles);

	}

	template<typename TDataType>
	void ConvertToTextureMesh<TDataType>::createMaterial()
	{
		std::shared_ptr<TextureMesh> texMesh = this->stateTextureMesh()->getDataPtr();
		if (texMesh->shapes().empty() | texMesh->meshDataPtr()->vertices().isEmpty())
			return;

		auto& TargetTriangles = texMesh->shapes()[0]->vertexIndex;
		auto& TargetPoints = texMesh->meshDataPtr()->vertices();
		// Assign Material;
		texMesh->shapes()[0]->material = MaterialManager::NewMaterial();

		//set Material
		dyno::CArray2D<dyno::Vec4f> texture(1, 1);

		auto diffusePath = this->varDiffuseTexture()->getValue().string();
		if (diffusePath.size())
		{
			if (loadTexture(diffusePath.c_str(), texture))
				texMesh->shapes()[0]->material->outTexColor()->getDataPtr()->assign(texture);
		}

		auto bumpPath = this->varNormalTexture()->getValue().string();
		if (bumpPath.size())
		{
			if (loadTexture(bumpPath.c_str(), texture))
				texMesh->shapes()[0]->material->outTexBump()->getDataPtr()->assign(texture);
		}

	}

	template<typename TDataType>
	void ConvertToTextureMesh<TDataType>::moveToCenter()
	{
		std::shared_ptr<TextureMesh> texMesh = this->stateTextureMesh()->getDataPtr();
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->inTopology()->getDataPtr());
		if (triSet == NULL)
			return;

		auto SourceTriangles = triSet->triangleIndices();
		auto SourcePoints = triSet->getPoints();

		auto& TargetTriangles = texMesh->shapes()[0]->vertexIndex;
		auto& TargetPoints = texMesh->meshDataPtr()->vertices();

		// Move To Center, use Transform
		Reduction<Coord> reduceBounding;

		auto& bounding = texMesh->shapes()[0]->boundingBox;
		Coord lo = reduceBounding.minimum(SourcePoints.begin(), SourcePoints.size());
		Coord hi = reduceBounding.maximum(SourcePoints.begin(), SourcePoints.size());

		bounding.v0 = lo;
		bounding.v1 = hi;

		if (this->varUseBoundingTransform()->getValue())
		{
			texMesh->shapes()[0]->boundingTransform.translation() = (lo + hi) / 2;
			DArray<Coord> iniPos;
			iniPos.assign(TargetPoints);

			cuExecute(TargetPoints.size(),
				ToCenter,
				iniPos,
				TargetPoints,
				(lo + hi) / 2
			);
		}
		else
		{
			texMesh->shapes()[0]->boundingTransform.translation() = Vec3f(0);
		}
	}

	DEFINE_CLASS(ConvertToTextureMesh);
}