#include <map>
#include "Object.h"

namespace dyno
{
static std::map< std::string, ClassInfo*> *classInfoMap = NULL;

IMPLEMENT_CLASS(Object)

#define BASE_ID 0

std::atomic_uint32_t Object::cId {BASE_ID + 1};

Object::Object()
{
	id = Object::generateObjectId();
}

bool Object::registerClass(ClassInfo* ci)
{
	if (!classInfoMap) {
		classInfoMap = new std::map< std::string, ClassInfo*>();
	}
	if (ci) {
		if (classInfoMap->find(ci->m_className) == classInfoMap->end()) {
			classInfoMap->insert(std::map< std::string, ClassInfo*>::value_type(ci->m_className, ci));
			// fprintf(stderr,"%s\n", ci->m_className.c_str());
		}
	}
	return true;
}
Object* Object::createObject(std::string name)
{
	std::map< std::string, ClassInfo*>::const_iterator iter = classInfoMap->find(name);
	if (classInfoMap->end() != iter) {
		return iter->second->createObject();
	}
	return NULL;
}

std::map< std::string, ClassInfo*>* Object::getClassMap()
{
	return classInfoMap;
}

ObjectId Object::baseId()
{
	return BASE_ID;
}

uint32_t Object::generateObjectId()
{
	return cId++;
}

bool Register(ClassInfo* ci)
{
	return Object::registerClass(ci);
}
}
