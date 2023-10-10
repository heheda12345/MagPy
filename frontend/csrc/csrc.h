#pragma once
#include <vector>

struct _object;
typedef _object PyObject;

namespace frontend_csrc {

struct Cache {
    PyObject *check_fn;
    PyObject *graph_fn;
    Cache *next;
};

typedef std::vector<Cache *> FrameCache;
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

} // namespace frontend_csrc
