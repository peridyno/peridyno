#define EKey ivec2
#define TKey ivec3
#define QKey ivec4

#define Coord vec3
#define Quad ivec4
#define Triangle ivec3
#define Edge ivec2
#define Tetrahedron ivec4
struct Hexahedron {
   int id[8];
};

#define Edg2Quad ivec2
#define Edg2Tri ivec2
#define Quad2Hex ivec2
#define Tri2Tet ivec2
#define Tri2Edg ivec3

#define EMPTY -1


bool compare_eq(in EKey a, in EKey b) {
   return a.x == b.x && a.y == b.y;
}
bool compare_eq(in TKey a, in TKey b) {
   return a.x == b.x && a.y == b.y && a.z == b.z;
}
bool compare_eq(in QKey a, in QKey b) {
   return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

bool compare_less(in EKey a, in EKey b) {
   return (a.x < b.x) || 
          (a.x == b.x && a.y < b.y);
}
bool compare_less(in TKey a, in TKey b) {
   return (a.x < b.x) || 
          (a.x == b.x && a.y < b.y) ||
          (a.x == b.x && a.y == b.y && a.z < b.z);
}
bool compare_less(in QKey a, in QKey b) {
   return (a.x < b.x) || 
          (a.x == b.x && a.y < b.y) ||
          (a.x == b.x && a.y == b.y && a.z < b.z) ||
          (a.x == b.x && a.y == b.y && a.z == b.z && a.w < b.w);
}

bool compare_large(in EKey a, in EKey b) {
   return (a.x > b.x) || 
          (a.x == b.x && a.y > b.y);
}
bool compare_large(in TKey a, in TKey b) {
   return (a.x > b.x) || 
          (a.x == b.x && a.y > b.y) ||
          (a.x == b.x && a.y == b.y && a.z > b.z);
}
bool compare_large(in QKey a, in QKey b) {
   return (a.x > b.x) || 
          (a.x == b.x && a.y > b.y) ||
          (a.x == b.x && a.y == b.y && a.z > b.z) ||
          (a.x == b.x && a.y == b.y && a.z == b.z && a.w > b.w);
}