#include<vector>

struct Cache {
    PyObject* check_fn;
    PyObject* graph_fn;
    Cache* next;
};

typedef std::vector<Cache*> FrameCache;
typedef std::vector<FrameCache> ProgramCache;