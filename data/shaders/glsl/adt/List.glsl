#ifndef ELEMENT
#define ELEMENT BufRef
#endif

#ifdef LIST_NAME
#define LIST(el) LIST_NAME
#define LIST_REF(el) LIST_REF_NAME
#define LIST_FUNC(el, name) CAT2(LIST_NAME, name)
#else
#define LIST(el) CAT(List, el)
#define LIST_REF(el) CAT(ListRef, el) 
#define LIST_FUNC(el, name) CAT2(List, CAT(el, name))
#endif

struct LIST(ELEMENT)
{
   uint max_size;
   ELEMENT buf;
   uint size;
   uint _padding;
};

layout(std430, buffer_reference, buffer_reference_align = 4) buffer LIST_REF(ELEMENT)
{
   LIST(ELEMENT) data;
};

ELEMENT LIST_FUNC(ELEMENT,atomicAdd)(LIST_REF(ELEMENT) self) {
   uint max_val = self.data.max_size;
   uint exp_val = 0;
   do {
      exp_val = self.data.size;
      if(exp_val == max_val) break;
   } while(atomicCompSwap(self.data.size, exp_val, exp_val + 1) != exp_val);
   return self.data.buf + (exp_val + 1);
}

ELEMENT LIST_FUNC(ELEMENT,get)(LIST(ELEMENT) self, in uint idx) {
   return self.buf + idx;
}
ELEMENT LIST_FUNC(ELEMENT,get)(LIST_REF(ELEMENT) self, in uint idx) {
   return LIST_FUNC(ELEMENT,get)(self.data, idx);
}

uint LIST_FUNC(ELEMENT,size)(LIST(ELEMENT) self) {
   return self.size;
}
uint LIST_FUNC(ELEMENT,size)(LIST_REF(ELEMENT) self) {
   return LIST_FUNC(ELEMENT,size)(self.data);
}

#undef LIST
#undef LIST_REF
#undef LIST_FUNC

#undef ELEMENT
#undef LIST_NAME
#undef LIST_REF_NAME