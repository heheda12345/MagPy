#pragma once
#include <map>
#include <string>
#include <vector>

struct _object;
typedef _object PyObject;

namespace frontend_csrc {

class NullObjectSingleton {
  public:
    static NullObjectSingleton &getInstance() {
        static NullObjectSingleton instance;
        return instance;
    }

    PyObject *getNullObject() { return this->null_object; }
    void setNullObject(PyObject *obj) { this->null_object = obj; }

  private:
    NullObjectSingleton() {}

    NullObjectSingleton(const NullObjectSingleton &) = delete;
    NullObjectSingleton &operator=(const NullObjectSingleton &) = delete;
    PyObject *null_object = nullptr;
};

struct Cache {
    PyObject *check_fn;
    PyObject *graph_fn;
    Cache *next;
    bool move_to_start;
};

struct FrameCache {
    std::vector<Cache *> caches;
    std::map<std::string, std::vector<std::string>> miss_locals;
};

typedef std::vector<FrameCache> ProgramCache;

// When not understanding an opcode, mark it as {-1, 0, stack_effect}
// if stack_effect > 0 or {-1, -stack_effect, 0, true, true} if stack_effect <
// 0, and update the opcode when needed
struct StackEffect {
    StackEffect(int read, int pop, int push, bool local_effect = false,
                bool global_effect = false)
        : read(read), write_old(pop), write_new(push),
          local_effect(local_effect), global_effect(global_effect) {}
    int read, write_old, write_new;
    bool local_effect, global_effect;
};
StackEffect stack_effect(int opcode, int oparg, int jump);
PyObject *parse_rangeiterobject(PyObject *self, PyObject *args);
PyObject *make_rangeiterobject(PyObject *self, PyObject *args);
PyObject *parse_mapproxyobject(PyObject *self, PyObject *args);
PyObject *parse_mapobject(PyObject *self, PyObject *args);
PyObject *parse_cell(PyObject *self, PyObject *args);
PyObject *set_cell(PyObject *self, PyObject *args);
PyObject *parse_type_obj(PyObject *self, PyObject *args);

} // namespace frontend_csrc
